# src/m3p/etl/build_trainset.py
import argparse
import json
import unicodedata
import yaml
import os
from pathlib import Path

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
    transcript.json の形式を正規化して、
      [{start_ms,end_ms,surface,punct,emo_id}, ...]
    という共通形にする。

    対応フォーマット:
    A) 既存形式: list[ {start_ms,end_ms,surface,punct?,emo_id?} ]
    B) faster-whisper形式:
        {
          "meta": {...},
          "segments": [
            {
              "segment_id": ...,
              "text": "...",
              "start_ms": ...,
              "end_ms": ...,
              "words": [
                {"word": "...", "start_ms": ..., "end_ms": ...}, ...
              ]
            },
            ...
          ]
        }
    """
    # A) 既存形式（すでにフラット）
    if isinstance(obj, list):
        return obj

    # B) faster-whisper形式
    if isinstance(obj, dict) and "segments" in obj:
        out = []
        for seg in obj.get("segments", []):
            seg_text = (seg.get("text") or "").strip()
            seg_words = seg.get("words") or []

            # セグメント末尾に句読点があれば punct として扱う
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
                        # まだ感情ラベルが無いので neutral で埋める（将来拡張可）
                        "emo_id": "neutral",
                    }
                )
        return out

    raise ValueError("Unexpected transcript.json format: must be list[...] or dict with 'segments' key")


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
    区間[t0, t1) を step_ms ごとにサンプルして mouth6 ID 配列に変換。
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


def main(cfg):
    paths = cfg["paths"]
    step_ms = cfg["step_ms"]

    os.makedirs(Path(paths["train_samples"]).parent, exist_ok=True)
    val_path = paths.get("val_samples")

    raw_tr = load_json(paths["transcript"])
    words = normalize_transcript(raw_tr)
    pose = load_json(paths["pose_timeline"])
    mouth_events = pose["timeline"]

    f_train = open(paths["train_samples"], "w", encoding="utf-8")
    f_val = open(val_path, "w", encoding="utf-8") if val_path else None

    train_cnt = 0
    val_cnt = 0
    is_first = True  # 最初の1サンプルだけ val に送る簡易split

    for ch in chunk_transcript(words):
        t0 = ch[0]["start_ms"]
        t1 = ch[-1]["end_ms"]
        kana = to_kana("".join([w.get("surface", "") + w.get("punct", "") for w in ch]))
        emo_id = ch[-1].get("emo_id", "neutral")

        steps = resample_steps(mouth_events, t0, t1, step_ms)
        line_obj = {
            "t0_ms": t0,
            "t1_ms": t1,
            "kana": kana,
            "emo_id": emo_id,
            "T": len(steps),
            "mouth_steps": steps,
        }
        line = json.dumps(line_obj, ensure_ascii=False) + "\n"

        if is_first and f_val:
            f_val.write(line)
            val_cnt += 1
            is_first = False
        else:
            f_train.write(line)
            train_cnt += 1

    f_train.close()
    if f_val:
        f_val.close()

    print(f"Wrote samples: train={train_cnt}, val={val_cnt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)

