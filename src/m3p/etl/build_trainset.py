import argparse
import json
import unicodedata
import yaml
import os
from pathlib import Path

###########################################################################
# 既存コード（to_kana, normalize_transcript, chunk_transcript, etc）
###########################################################################

MOUTHS = ["close", "a", "i", "u", "e", "o"]
M2ID = {m: i for i, m in enumerate(MOUTHS)}

# かな化（pykakasiがあれば使用、なければNFKC）
try:
    from pykakasi import kakasi
    _k = kakasi()
    _k.setMode("J", "H")
    _k.setMode("K", "H")
    _k.setMode("H", "H")
    _conv = _k.getConverter()

    def to_kana(s: str) -> str:
        try:
            return _conv.do(s.strip())
        except Exception:
            return unicodedata.normalize("NFKC", s)
except Exception:
    def to_kana(s: str) -> str:
        return unicodedata.normalize("NFKC", s)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_transcript(obj):
    """
    transcript.json を正規化して、
      [{start_ms,end_ms,surface,punct,emo_id}, ...]
    の共通形式にする。
    """
    # --- 既存形式: list そのまま
    if isinstance(obj, list):
        return obj

    # --- faster-whisper 形式
    if isinstance(obj, dict) and "segments" in obj:
        out = []
        for seg in obj.get("segments", []):
            seg_text = (seg.get("text") or "").strip()
            seg_words = seg.get("words") or []

            punct_char = ""
            if seg_text and seg_text[-1] in "。！？!?":
                punct_char = seg_text[-1]

            for idx, w in enumerate(seg_words):
                surface = (w.get("word") or "").strip()
                if not surface:
                    continue
                start_ms = int(round(w.get("start_ms", seg.get("start_ms", 0))))
                end_ms = int(round(w.get("end_ms", seg.get("end_ms", start_ms))))
                punct = punct_char if (punct_char and idx == len(seg_words) - 1) else ""
                out.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "surface": surface,
                        "punct": punct,
                        "emo_id": "neutral",
                    }
                )
        return out

    raise ValueError("Unexpected transcript.json format")


def chunk_transcript(words, min_ms=300, max_ms=800):
    """300〜800ms程度でチャンクする。 punct でも切る。"""
    cur = []
    start = None
    end = None
    for w in words:
        if start is None:
            start = w["start_ms"]
        cur.append(w)
        end = w["end_ms"]
        dur = end - start
        punct = w.get("punct", "")

        if dur >= min_ms and (dur >= max_ms or punct in ["。", "！", "？", ".", "!", "?", "\n"]):
            yield cur
            cur = []
            start = None
            end = None

    if cur:
        yield cur


def resample_steps(mouth_events, t0, t1, step_ms):
    """
    mouth_events: [{"t_ms":..., "mouth6":...}, ...]
    区間[t0, t1) を step_ms ごとに mouth6 ID 配列に変換。
    """
    ev = sorted(mouth_events, key=lambda x: x["t_ms"])
    if ev:
        ev.append({"t_ms": 10**12, "mouth6": ev[-1]["mouth6"]})
    else:
        ev = [{"t_ms": 0, "mouth6": "close"}, {"t_ms": 10**12, "mouth6": "close"}]

    def mouth_at(t):
        prev = ev[0]["mouth6"]
        for e in ev[1:]:
            if t < e["t_ms"]:
                return prev
            prev = e["mouth6"]
        return prev

    T = max(1, round((t1 - t0) / step_ms))
    out = []
    for i in range(T):
        m = mouth_at(t0 + i * step_ms)
        out.append(M2ID.get(m, 0))
    return out


###########################################################################
# Day2 manifest モード用：追加ユーティリティ
###########################################################################

def load_manifest(path: str):
    """
    manifest YAML を読んで、
      - 全セッション dict
      - train セッション list
      - val セッション list
    を返す
    """
    with open(path, "r", encoding="utf-8") as f:
        m = yaml.safe_load(f)

    sessions_by_id = {s["id"]: s for s in m["sessions"]}
    train_ids = m["split"]["train_ids"]
    val_ids = m["split"]["val_ids"]

    train_sess = [sessions_by_id[x] for x in train_ids]
    val_sess = [sessions_by_id[x] for x in val_ids]
    return train_sess, val_sess


def build_item_from_session(sess: dict, step_ms: int):
    """
    manifest 内の1セッション定義から M3’ JSONL の1行を作る。
    """
    transcript = load_json(sess["transcript"])
    pose = load_json(sess["pose_timeline"])

    words = normalize_transcript(transcript)
    mouth_events = pose["timeline"]

    # Day2最小セットなので「1セッション＝1サンプル」
    t0 = words[0]["start_ms"]
    t1 = words[-1]["end_ms"]

    text = "".join([w["surface"] + w.get("punct", "") for w in words])
    kana = to_kana(text)
    emo_id = sess["emo_id"]

    steps = resample_steps(mouth_events, t0, t1, step_ms)

    return {
        "id": sess["id"],
        "base_id": sess.get("base_id"),
        "emo_id": emo_id,
        "kana": kana,
        "T": len(steps),
        "mouth_steps": steps,
        "t0_ms": t0,
        "t1_ms": t1
    }


def build_jsonl_from_sessions(sessions, out_path: Path, step_ms: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        for sess in sessions:
            item = build_item_from_session(sess, step_ms)
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


###########################################################################
# 既存（単一ファイル）モード ＋ manifest モードの2段構え main
###########################################################################

def main(cfg):
    """
    config.yaml に manifest が指定されていれば manifest モード。
    無ければ旧仕様の単一ファイルモードで動作。
    """
    step_ms = cfg["step_ms"]

    # -----------------------------
    # (A) manifest モード（Day2以降）
    # -----------------------------
    manifest = cfg["paths"].get("manifest")
    if manifest:
        print(f"[build_trainset] manifest モード: {manifest}")
        train_sess, val_sess = load_manifest(manifest)

        out_train = Path(cfg["paths"]["train_samples"])
        out_val = Path(cfg["paths"]["val_samples"])

        build_jsonl_from_sessions(train_sess, out_train, step_ms)
        build_jsonl_from_sessions(val_sess, out_val, step_ms)

        print(f"Wrote: train={len(train_sess)}, val={len(val_sess)}")
        return

    # -----------------------------
    # (B) 既存（単一ファイル）モード
    # -----------------------------
    print("[build_trainset] 単一ファイルモード（従来動作）")

    paths = cfg["paths"]
    raw_tr = load_json(paths["transcript"])
    words = normalize_transcript(raw_tr)
    pose = load_json(paths["pose_timeline"])
    mouth_events = pose["timeline"]

    Path(paths["train_samples"]).parent.mkdir(parents=True, exist_ok=True)
    f_train = open(paths["train_samples"], "w", encoding="utf-8")
    f_val = open(paths["val_samples"], "w", encoding="utf-8")

    train_cnt = 0
    val_cnt = 0
    is_first = True

    for ch in chunk_transcript(words):
        t0 = ch[0]["start_ms"]
        t1 = ch[-1]["end_ms"]
        kana = to_kana("".join([w.get("surface", "") + w.get("punct", "") for w in ch]))
        emo_id = ch[-1].get("emo_id", "neutral")

        steps = resample_steps(mouth_events, t0, t1, step_ms)
        obj = {
            "t0_ms": t0,
            "t1_ms": t1,
            "kana": kana,
            "emo_id": emo_id,
            "T": len(steps),
            "mouth_steps": steps,
        }
        line = json.dumps(obj, ensure_ascii=False) + "\n"

        if is_first:
            f_val.write(line)
            val_cnt += 1
            is_first = False
        else:
            f_train.write(line)
            train_cnt += 1

    f_train.close()
    f_val.close()
    print(f"Wrote samples: train={train_cnt}, val={val_cnt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
