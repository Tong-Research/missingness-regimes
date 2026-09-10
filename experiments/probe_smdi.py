"""SMDI (prereg/SMDI.md): does the single-model diagnostic of smdi (Weberpals et al., JAMIA Open 2024) give the same
answer as measuring the two predictor blocks separately?

smdi's Group 2 diagnostic fits ONE model per column, predicting its missingness indicator from covariates AND the other
columns' missingness indicators together, and reads the answer off variable importance. This script fits that single
model and, on the same panels and splits as RRSTRUCT, compares its BLOCK importance with the standalone AUROC of each
block fitted alone. Panels, splits, representative columns and the tree are RRSTRUCT's, unchanged.

Per panel: auroc_values, auroc_panels (each block alone), auroc_both, and for each of 3 seeds the permutation
importance of each block inside the single model (summed over its columns, floored at 0), giving imp_share_values."""
import argparse, csv, numpy as np
from probe_unseen import load_full
from probe_common import tree
from probe_explore import keepcols
from sklearn.metrics import roc_auc_score

def auc(y, s): return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan

def main():
    a = argparse.ArgumentParser(); a.add_argument("--dataset", required=True); a.add_argument("--out", required=True); a = a.parse_args()
    Xd, Xh = load_full(a.dataset); keep = ~(np.isnan(Xd).all(0) & np.isnan(Xh).all(0)); Xd, Xh = Xd[:, keep], Xh[:, keep]
    rng = np.random.default_rng(0); cap = 60_000
    if len(Xd) > cap: Xd = Xd[rng.choice(len(Xd), cap, replace=False)]
    if len(Xh) > cap: Xh = Xh[rng.choice(len(Xh), cap, replace=False)]
    Md, Mh = np.isnan(Xd), np.isnan(Xh); rate = Md.mean(0); p = Xd.shape[1]
    mc = [j for j in range(p) if 0.01 <= rate[j] <= 0.99]; full = [j for j in range(p) if rate[j] < 0.01]
    clus = {}
    for j in mc:
        for k in clus:
            i_, u_ = (Md[:, j] & Md[:, k]).sum(), (Md[:, j] | Md[:, k]).sum()
            if u_ and i_ / u_ >= 0.9: clus[j] = clus[k]; break
        else: clus[j] = j
    panels = {}
    for j in mc: panels.setdefault(clus[j], []).append(j)
    reps = [sorted(v, key=lambda j: abs(rate[j] - np.median(rate[v])))[0] for v in panels.values()]
    Pd, Ph = (~Md[:, reps]).astype(float), (~Mh[:, reps]).astype(float); B = len(reps); rows = []
    print(f"{a.dataset}: panels {B}, always-observed {len(full)}", flush=True)
    if not full: print("  no always-observed columns; nothing to compare"); 
    for b in range(B):
        yd, yh = 1 - Pd[:, b], 1 - Ph[:, b]
        if not (0.01 <= yd.mean() <= 0.99 and 0.01 <= yh.mean() <= 0.99) or len(np.unique(yh)) < 2 or not full: continue
        oth = [c for c in range(B) if c != b]
        av = auc(yh, tree().fit(Xd[:, full][:, keepcols(Xd[:, full])], yd).predict_proba(Xh[:, full][:, keepcols(Xd[:, full])])[:, 1])
        ap = auc(yh, tree().fit(Pd[:, oth], yd).predict_proba(Ph[:, oth])[:, 1]) if oth else np.nan
        Fd = np.hstack([Xd[:, full], Pd[:, oth]]); Fh = np.hstack([Xh[:, full], Ph[:, oth]]); kc = keepcols(Fd)
        mdl = tree().fit(Fd[:, kc], yd); base = auc(yh, mdl.predict_proba(Fh[:, kc])[:, 1]); nv = len(full)
        for seed in range(3):
            r2 = np.random.default_rng(seed); imp = np.zeros(len(kc))
            for i in range(len(kc)):
                Fp = Fh.copy(); Fp[:, kc[i]] = Fp[r2.permutation(len(Fp)), kc[i]]
                imp[i] = max(0.0, base - auc(yh, mdl.predict_proba(Fp[:, kc])[:, 1]))
            iv = imp[[i for i, c in enumerate(kc) if c < nv]].sum(); ip = imp[[i for i, c in enumerate(kc) if c >= nv]].sum()
            rows.append(dict(dataset=a.dataset, panel=b, seed=seed, auroc_values=av, auroc_panels=ap, auroc_both=base,
                             imp_values=iv, imp_panels=ip, imp_share_values=(iv / (iv + ip)) if (iv + ip) > 0 else np.nan, n_panels=B, n_full=len(full)))
        s = np.mean([r["imp_share_values"] for r in rows[-3:]])
        print(f"  panel {b:3d} values {av:.3f} panels {ap:.3f} both {base:.3f} | importance share to values {s:.3f}", flush=True)
    if not rows: rows = [dict(dataset=a.dataset, panel=-1, seed=-1, auroc_values=np.nan, auroc_panels=np.nan, auroc_both=np.nan, imp_values=np.nan, imp_panels=np.nan, imp_share_values=np.nan, n_panels=B, n_full=len(full))]
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
if __name__ == "__main__": main()
