# src/m3p/infer/jsonl_infer.py
import argparse, yaml, json, torch, time, unicodedata, os
from m3p.data.dataset import VOCAB
from m3p.model.t2mouth import Text2Mouth

MOUTHS = ["close","a","i","u","e","o"]

def to_kana(s):
    try:
        from pykakasi import kakasi
        _k=kakasi(); _k.setMode("J","H"); _k.setMode("K","H"); _k.setMode("H","H"); conv=_k.getConverter()
        return conv.do(s.strip())
    except:
        return unicodedata.normalize("NFKC", s)

C2I = {c:i+1 for i,c in enumerate(
    list("ぁあぃいぅうぇえぉおかがきぎくぐけげこご"
         "さざしじすずせぜそぞただちぢっつづてでとど"
         "なにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
         "まみむめもやゃゆゅよょらりるれろわゐゑをんー、。！？.!?\n 　"))}
def text_to_ids(s, max_len=160):
    ids=[C2I.get(c,0) for c in s[:max_len]]; return ids, len(ids)

def write_mouth_tl(tl, path, step_ms):
    obj={"meta":{"version":"1.0.0","generator":"m3p",  # ← generator名だけ変更
                 "created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "time_unit":"ms","step_ms":step_ms},
         "timeline":tl}
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,ensure_ascii=False,indent=2)

def main(cfg, in_path, out_path):
    dev=cfg["device"] if torch.cuda.is_available() else "cpu"; step_ms=cfg["step_ms"]
    out_dir=cfg["paths"]["out_dir"]
    m=Text2Mouth(VOCAB, **cfg["model"]).to(dev)
    m.load_state_dict(torch.load(f"{out_dir}/ckpt/best.pt", map_location=dev))
    m.eval()

    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            js = json.loads(line)
            kana = js["kana"]; T = js["T"]
            x_ids, L = text_to_ids(kana)
            x=torch.tensor([x_ids]).long().to(dev)
            mask=torch.zeros_like(x).bool(); mask[0,:L]=1

            with torch.no_grad():
                logits=m(x,mask,[T]).squeeze(0)
                pred_ids=logits.argmax(-1).cpu().tolist()

            js["mouth_steps_pred"] = pred_ids
            f_out.write(json.dumps(js, ensure_ascii=False) + "\n")
    print(f"Wrote: {out_path}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--in",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config,"r",encoding="utf-8"))
    main(cfg, getattr(a, "in"), a.out)

