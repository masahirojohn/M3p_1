import argparse, yaml, json, os
from m3prime.metrics import summarize_sample
import numpy as np

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main(cfg, gt_path, pred_path, report_path):
    step_ms = cfg["step_ms"]
    gts = load_jsonl(gt_path)
    preds = load_jsonl(pred_path)

    all_metrics = []
    for gt, pred in zip(gts, preds):
        y_true = gt["mouth_steps"]
        y_pred = pred["mouth_steps_pred"]
        metrics = summarize_sample(y_true, y_pred, step_ms)
        all_metrics.append(metrics)

    f1 = np.mean([a["f1_macro"] for a in all_metrics])
    sw = np.mean([a["switch_per_sec"] for a in all_metrics])
    mae = np.mean([a["timing_mae_ms"] for a in all_metrics])

    summary = {
        "f1_macro": float(f1),
        "switch_per_sec": float(sw),
        "timing_mae_ms": float(mae)
    }

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.gt, args.pred, args.report)
