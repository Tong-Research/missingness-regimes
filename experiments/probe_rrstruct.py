"""RRSTRUCT (prereg/RRSTRUCT.md): decompose indicator-indicator dependence into a low-dimensional latent regime
(common cause) versus direct edges, and test whether the observed values explain it.

Panels = Jaccard >= 0.9 clusters of co-missing columns (as in Result 209); a panel's presence is its representative
column's observed indicator (representative = median missing rate within the cluster). Panels are defined on the DEV
half; every model is FIT on dev and evaluated on HOLDOUT. For each panel b with base rate in [0.01, 0.99] on both
halves, AUROC of predicting b's presence from:
  values      always-observed columns' values (tree)                  -- can R be explained by X?
  panels_tree other panels' presences (tree, unrestricted)            -- the reference dependence
  panels_l1   other panels' presences (L1 logistic = Ising neighbourhood selection) -- pairwise direct edges
  lca_<K>     other panels, via a K-class Bernoulli mixture posterior -- pure common cause, no direct edges
  both        values + other panels (tree)
Bernoulli mixtures are fitted on the dev panel matrix by EM (seed 0, 200 iterations, Laplace 1e-3), K in
{1,2,5,10,20,50,100}. With --sites <db> the class assignment is compared with the hospital/unit id."""
import argparse, csv, pathlib, numpy as np
from probe_unseen import load_full
from probe_common import tree
from probe_explore import keepcols
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
KS = [1, 2, 5, 10, 20, 50, 100]
C = pathlib.Path.home() / ".cache/phd-matrices"

def fit_bmm(P, K, seed=0, iters=200, tol=1e-7, eps=1e-3):
    rng = np.random.default_rng(seed); n, B = P.shape
    theta = rng.uniform(0.25, 0.75, (K, B)); pi = np.full(K, 1.0 / K); prev = -np.inf
    for _ in range(iters):
        lg = P @ np.log(theta.T + 1e-12) + (1 - P) @ np.log(1 - theta.T + 1e-12) + np.log(pi + 1e-12)
        m = lg.max(1, keepdims=True); w = np.exp(lg - m); s = w.sum(1, keepdims=True); r = w / s
        ll = float((np.log(s).ravel() + m.ravel()).mean())
        pi = r.mean(0) + 1e-12; pi /= pi.sum(); theta = (r.T @ P + eps) / (r.sum(0)[:, None] + 2 * eps)
        if ll - prev < tol: break
        prev = ll
    return pi, theta, prev

def lca_posterior(pi, theta, P, drop=None):
    idx = [j for j in range(P.shape[1]) if j != drop] if drop is not None else list(range(P.shape[1]))
    lg = P[:, idx] @ np.log(theta[:, idx].T + 1e-12) + (1 - P[:, idx]) @ np.log(1 - theta[:, idx].T + 1e-12) + np.log(pi + 1e-12)
    m = lg.max(1, keepdims=True); w = np.exp(lg - m); return w / w.sum(1, keepdims=True)

def auc(y, s): return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan

def main():
    a = argparse.ArgumentParser(); a.add_argument("--dataset", required=True); a.add_argument("--sites", default=""); a.add_argument("--mirror", action="store_true", help="swap the halves: fit on holdout, evaluate on dev (RRMIRROR)")
    a.add_argument("--drop-prefix", default="", help="comma-separated column-name prefixes to remove before anything else (NOTREAT: vent_,inf_,rrt_)")
    a.add_argument("--out", required=True); a = a.parse_args()
    Xd, Xh = load_full(a.dataset)
    if a.mirror: Xd, Xh = Xh, Xd
    if a.drop_prefix:
        # NOTREAT (prereg/NOTREAT.md). A missing treatment field means the treatment was not given, which is a
        # different object from an unordered measurement. Dropping those columns tests whether the dissociation
        # survives without them. Done BEFORE the all-empty filter so column indices stay aligned with cols.
        zc = np.load(C / (f"{a.dataset}.npz"), allow_pickle=True)
        cols = [str(c) for c in zc["cols"]]
        pref = tuple(x for x in a.drop_prefix.split(",") if x)
        keepj = np.array([not c.startswith(pref) for c in cols])
        assert len(cols) == Xd.shape[1], f"cols {len(cols)} != matrix width {Xd.shape[1]}"
        print(f"drop-prefix {pref}: {int((~keepj).sum())} of {len(cols)} columns removed", flush=True)
        Xd, Xh = Xd[:, keepj], Xh[:, keepj]
    keep = ~(np.isnan(Xd).all(0) & np.isnan(Xh).all(0)); Xd, Xh = Xd[:, keep], Xh[:, keep]
    cap = 60_000; rng = np.random.default_rng(0)
    if len(Xd) > cap: Xd = Xd[rng.choice(len(Xd), cap, replace=False)]
    idxh = rng.choice(len(Xh), min(cap, len(Xh)), replace=False); Xh_full = Xh; Xh = Xh[idxh]
    Md, Mh = np.isnan(Xd), np.isnan(Xh); rate = Md.mean(0); p = Xd.shape[1]
    mc = [j for j in range(p) if 0.01 <= rate[j] <= 0.99]; full = [j for j in range(p) if rate[j] < 0.01]
    clus = {}
    for j in mc:
        for k in clus:
            inter = (Md[:, j] & Md[:, k]).sum(); uni = (Md[:, j] | Md[:, k]).sum()
            if uni and inter / uni >= 0.9: clus[j] = clus[k]; break
        else: clus[j] = j
    panels = {}
    for j in mc: panels.setdefault(clus[j], []).append(j)
    reps = [sorted(v, key=lambda j: abs(rate[j] - np.median(rate[v])))[0] for v in panels.values()]
    Pd, Ph = (~Md[:, reps]).astype(float), (~Mh[:, reps]).astype(float); B = len(reps)
    rows = []; print(f"{a.dataset}: n_dev {len(Pd):,} p {p} cols-with-missing {len(mc)} panels {B} always-observed {len(full)}", flush=True)
    models = {}
    for K in KS:
        if K > 2 ** B: continue
        pi, th, ll = fit_bmm(Pd, K); models[K] = (pi, th)
        lgh = Ph @ np.log(th.T + 1e-12) + (1 - Ph) @ np.log(1 - th.T + 1e-12) + np.log(pi + 1e-12)
        m = lgh.max(1, keepdims=True); hll = float((np.log(np.exp(lgh - m).sum(1)) + m.ravel()).mean())
        rows.append(dict(dataset=a.dataset, panel=-1, kind="fit", model=f"lca_{K}", value=hll, n_panels=B, n_cols_missing=len(mc), n_full=len(full), n_dev=len(Pd), n_holdout=len(Ph)))
        print(f"  K={K:3d} holdout mixture log-lik/row {hll:.4f}", flush=True)
    for b in range(B):
        yd, yh = 1 - Pd[:, b], 1 - Ph[:, b]          # predict MISSING, as in Result 209
        if not (0.01 <= yd.mean() <= 0.99 and 0.01 <= yh.mean() <= 0.99) or len(np.unique(yh)) < 2: continue
        oth = [c for c in range(B) if c != b]; res = {}
        if full: kc = keepcols(Xd[:, full]); res["values"] = auc(yh, tree().fit(Xd[:, full][:, kc], yd).predict_proba(Xh[:, full][:, kc])[:, 1])
        res["panels_tree"] = auc(yh, tree().fit(Pd[:, oth], yd).predict_proba(Ph[:, oth])[:, 1])
        res["panels_l1"] = auc(yh, LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000).fit(Pd[:, oth], yd).predict_proba(Ph[:, oth])[:, 1]) if len(oth) else np.nan
        for K, (pi, th) in models.items(): res[f"lca_{K}"] = auc(yh, lca_posterior(pi, th, Ph, drop=b) @ (1 - th[:, b]))
        if full:
            kc = keepcols(np.hstack([Xd[:, full], Pd[:, oth]])); res["both"] = auc(yh, tree().fit(np.hstack([Xd[:, full], Pd[:, oth]])[:, kc], yd).predict_proba(np.hstack([Xh[:, full], Ph[:, oth]])[:, kc])[:, 1])
        for k, v in res.items(): rows.append(dict(dataset=a.dataset, panel=b, kind="auroc", model=k, value=v, n_panels=B, n_cols_missing=len(mc), n_full=len(full), n_dev=len(Pd), n_holdout=len(Ph)))
        print(f"  panel {b:3d} rate {yh.mean():.3f} " + " ".join(f"{k}={v:.3f}" for k, v in res.items() if k in ("values", "panels_tree", "panels_l1", "lca_10", "lca_100", "both")), flush=True)
    if a.sites:
        from sklearn.metrics import adjusted_mutual_info_score as ami
        g = np.asarray(np.load(C / f"{a.sites}_sites.npz", allow_pickle=True)["g"]).ravel()
        yall = np.asarray(np.load(C / f"cand_{a.sites}_regime.npz", allow_pickle=True)["y"]).ravel()
        from probe_unseen import dev_mask, mimic_dev
        # The regime matrix carries its OWN hash split (dev_mask on its own name), which is what load_full uses.
        # Using mimic_split.dev_mask here mismatched by 51 rows and crashed MIMIC-IV (2026-09-06 20:15).
        dm = dev_mask(f"cand_{a.sites}_regime", len(yall))
        if a.mirror: dm = ~dm
        gh = g[~dm][idxh]
        for K, (pi, th) in models.items():
            if K < 2: continue
            z = lca_posterior(pi, th, Ph).argmax(1); maj = np.bincount(gh).max() / len(gh)
            acc = np.mean([np.bincount(gh[z == c]).max() if (z == c).any() else 0 for c in range(K)]) * 0 + sum(np.bincount(gh[z == c]).max() for c in range(K) if (z == c).any()) / len(gh)
            rows.append(dict(dataset=a.dataset, panel=-1, kind="sites", model=f"lca_{K}", value=float(ami(gh, z)), n_panels=B, n_cols_missing=len(mc), n_full=len(full), n_dev=len(Pd), n_holdout=len(Ph)))
            rows.append(dict(dataset=a.dataset, panel=-2, kind="sites_purity", model=f"lca_{K}", value=float(acc / maj), n_panels=B, n_cols_missing=len(mc), n_full=len(full), n_dev=len(Pd), n_holdout=len(Ph)))
            print(f"  K={K:3d} vs {a.sites} sites: AMI {ami(gh, z):.3f}, purity/majority {acc/maj:.2f}", flush=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
if __name__ == "__main__": main()
