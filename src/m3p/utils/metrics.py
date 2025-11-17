import json, numpy as np
from sklearn.metrics import f1_score

def switch_per_second(ids, step_ms):
    if len(ids)<=1: return 0.0
    sw=sum(1 for i in range(1,len(ids)) if ids[i]!=ids[i-1])
    dur_ms=len(ids)*step_ms
    return (sw / max(1e-6, dur_ms/1000.0))

def timing_mae_ms(y_true_ids, y_pred_ids, step_ms):
    # 切替境界の位置（インデックス）を抽出して MAE
    def edges(ids):
        return [i for i in range(1,len(ids)) if ids[i]!=ids[i-1]]
    et, ep = np.array(edges(y_true_ids)), np.array(edges(y_pred_ids))
    if len(et)==0 or len(ep)==0: 
        # どちらかに境界がほぼ無ければ長さ誤差を近似指標に
        return abs(len(y_true_ids)-len(y_pred_ids))*step_ms
    # 長さに合わせて線形スケールで対応づけ
    k = max(1, min(len(et), len(ep)))
    et_res = np.linspace(0, len(y_true_ids), k, endpoint=False)
    ep_res = np.linspace(0, len(y_pred_ids), k, endpoint=False)
    return float(np.mean(np.abs(et_res - ep_res))) * step_ms

def summarize_sample(y_true_ids, y_pred_ids, step_ms):
    f1 = f1_score(y_true_ids, y_pred_ids, average="macro")
    sw = switch_per_second(y_pred_ids, step_ms)
    mae= timing_mae_ms(y_true_ids, y_pred_ids, step_ms)
    return {"f1_macro": float(f1), "switch_per_sec": float(sw), "timing_mae_ms": float(mae)}
