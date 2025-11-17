import argparse, yaml, os, json, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from m3prime.dataset import MouthDataset, collate, VOCAB
from m3prime.model_tiny import Text2Mouth

def main(cfg):
    torch.manual_seed(cfg["seed"])
    dev = cfg["device"] if torch.cuda.is_available() else "cpu"
    out = cfg["paths"]["out_dir"]; os.makedirs(f"{out}/ckpt", exist_ok=True); os.makedirs(f"{out}/metrics", exist_ok=True)
    ds = MouthDataset(cfg["paths"]["train_samples"])
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate)

    m = Text2Mouth(VOCAB, **cfg["model"]).to(dev)
    opt = optim.AdamW(m.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ce  = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=cfg["train"]["label_smoothing"])

    best=float("inf"); no_improve=0
    for ep in range(1, cfg["train"]["epochs"]+1):
        m.train(); tot=0; steps=0
        for B in dl:
            x=B["x"].to(dev); mask=B["mask"].to(dev); y=B["y"].to(dev); T=B["T_list"]
            logits=m(x,mask,T); loss=0.0; n=0
            for i,t in enumerate(T): loss += ce(logits[i,:t], y[i,:t]); n+=t
            loss/=max(1,n); opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item(); steps+=1
        avg=tot/max(1,steps)
        print(f"[ep {ep}] loss={avg:.4f}")
        # pseudo-valid: use train loss trend（最小版）
        if avg < best-1e-4:
            best=avg; no_improve=0
            torch.save(m.state_dict(), f"{out}/ckpt/best.pt")
        else:
            no_improve+=1
            if no_improve>=cfg["train"]["early_stop_patience"]:
                break

    with open(f"{out}/metrics/train_summary.json","w") as f:
        json.dump({"best_train_loss":best}, f, indent=2)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    args=ap.parse_args(); cfg=yaml.safe_load(open(args.config,"r",encoding="utf-8"))
    main(cfg)
