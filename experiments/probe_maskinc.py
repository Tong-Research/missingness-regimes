"""MASKINC (prereg/MASKINC.md): does the explicit mask start to pay once the values can no longer reveal the regime?
Result 209: adding the mask to a native tree gains 0.000 AUPRC with all columns present. Under the latent-regime
account (X reveals Z, so the mask is redundant) the gain must GROW as value columns are withheld.

Holdout half, cap 60,000, 3-fold stratified CV seed 0, HCAL tree config. For q in {1.0, 0.75, 0.5, 0.25, 0.1} keep a
random subset of ceil(q*p) value columns (3 draws per q, seeds 0-2; q=1.0 once), and compare
  values            tree on the retained value columns
  values+mask_full  tree on retained values + ALL p indicators   (does the mask carry what the withheld values did?)
  values+mask_kept  tree on retained values + the retained columns' indicators (realistic ablation)
  mask_full         tree on all p indicators alone (constant across q; reported once)
gain_full = AUPRC(values+mask_full) - AUPRC(values); gain_kept likewise."""
import argparse, csv, numpy as np
from probe_common import load, tree
from probe_explore import keepcols, oof
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap_
QS = [1.0, 0.75, 0.5, 0.25, 0.1]

def main():
    a = argparse.ArgumentParser(); a.add_argument("--dataset", required=True); a.add_argument("--slice", default="holdout"); a.add_argument("--out", required=True); a = a.parse_args()
    X, y = load(a.dataset, a.slice, 60_000); n, p = X.shape; M = np.isnan(X).astype(float)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y)); rows = []
    mask_only = ap_(y, oof(M, y, folds)); prev = y.mean()
    print(f"{a.dataset}: n {n:,} p {p} prevalence {prev:.4f} mask-only AUPRC {mask_only:.4f}", flush=True)
    rows.append(dict(dataset=a.dataset, slice=a.slice, q=0.0, draw=-1, k=0, arm="mask_full", auprc=mask_only, prevalence=prev, n=n, p=p))
    for q in QS:
        k = max(1, int(np.ceil(q * p))); draws = 1 if q == 1.0 else 3
        for d in range(draws):
            cols = np.arange(p) if q == 1.0 else np.random.default_rng(d).choice(p, k, replace=False)
            V = X[:, cols]; arms = {"values": V, "values+mask_full": np.hstack([V, M]), "values+mask_kept": np.hstack([V, M[:, cols]])}
            r = {name: ap_(y, oof(F, y, folds)) for name, F in arms.items()}
            for name, v in r.items(): rows.append(dict(dataset=a.dataset, slice=a.slice, q=q, draw=d, k=k, arm=name, auprc=v, prevalence=prev, n=n, p=p))
            print(f"  q={q:.2f} draw {d} k={k}: values {r['values']:.4f} +mask_full {r['values+mask_full']:.4f} (gain {r['values+mask_full']-r['values']:+.4f}) +mask_kept {r['values+mask_kept']:.4f} (gain {r['values+mask_kept']-r['values']:+.4f})", flush=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
if __name__ == "__main__": main()
