"""REGIME (prereg/REGIME.md): is the ordering regime hospital-specific, and is regime shift a cause of cross-hospital loss?
Rows of cand_<db>_regime are aligned with <db>_sites (asserted on y), which carries the site id g.
  --mode site:     3-fold CV top-1 accuracy of a tree predicting the site from the mask only, the mean-filled values
                   only, and both; majority-class baseline. All sites, and the 20 largest.
  --mode transfer --site h: T = rows of site h, S = the rest. in_site: 3-fold CV native tree within T. transfer: tree
                   on S -> T. adapted: S re-masked with masks sampled from T (a source value is dropped where the
                   sampled target mask is missing; nothing is ever added) -> T. mask_only variants of transfer and
                   in_site. regime_distance = total-variation distance between S's and T's per-column missing rates."""
import argparse, csv, pathlib, numpy as np
from probe_common import tree, SimpleImputer
from probe_explore import keepcols, oof
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap_
C = pathlib.Path.home() / ".cache/phd-matrices"

def load(db):
    zr, zs = np.load(C / f"cand_{db}_regime.npz", allow_pickle=True), np.load(C / f"{db}_sites.npz", allow_pickle=True)
    X, y, g = np.asarray(zr["X"], float), np.asarray(zr["y"]).ravel().astype(int), np.asarray(zs["g"]).ravel()
    assert (y == np.asarray(zs["y"]).ravel().astype(int)).all(); return X[:, ~np.isnan(X).all(0)], y, g

def fit_pred(Ftr, ytr, Fte):
    kc = keepcols(Ftr); return tree().fit(Ftr[:, kc], ytr).predict_proba(Fte[:, kc])[:, 1]

def main():
    a = argparse.ArgumentParser(); a.add_argument("--db", required=True); a.add_argument("--mode", choices=["site", "transfer"], required=True); a.add_argument("--site", type=int, default=-1); a.add_argument("--out", required=True); a = a.parse_args()
    X, y, g = load(a.db); M = np.isnan(X); rows = []
    if a.mode == "site":
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        Xi = SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(X)
        for label, keep in [("all", np.ones(len(g), bool)), ("top20", np.isin(g, np.argsort(np.bincount(g))[::-1][:20]))]:
            gg = g[keep]; base = np.bincount(gg).max() / len(gg); folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xi[keep], gg))
            for feat, F in [("mask", M[keep].astype(float)), ("values", Xi[keep]), ("both", np.hstack([Xi[keep], M[keep].astype(float)]))]:
                pred = np.zeros(len(gg), int)
                for tr, te in folds: kc = keepcols(F[tr]); pred[te] = H(max_depth=3, learning_rate=0.1, max_iter=150, random_state=0, early_stopping=False).fit(F[tr][:, kc], gg[tr]).predict(F[te][:, kc])
                acc = float((pred == gg).mean()); rows.append(dict(db=a.db, subset=label, n_sites=len(np.unique(gg)), n=len(gg), features=feat, accuracy=acc, majority=base, ratio=acc / base)); print(rows[-1], flush=True)
    else:
        T = g == a.site; S = ~T; yT, yS = y[T], y[S]; rng = np.random.default_rng(a.site)
        if yT.sum() < 50 or (~T).sum() == 0: rows.append(dict(db=a.db, site=a.site, n_T=int(T.sum()), pos_T=int(yT.sum()), note="not applicable")); 
        else:
            rate_S, rate_T = M[S].mean(0), M[T].mean(0); dist = float(np.abs(rate_S - rate_T).mean())
            XS, XT = X[S], X[T]; folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(XT, yT))
            in_site = ap_(yT, oof(XT, yT, folds)); transfer = ap_(yT, fit_pred(XS, yS, XT))
            Xad = XS.copy(); Xad[M[T][rng.integers(0, T.sum(), S.sum())]] = np.nan; adapted = ap_(yT, fit_pred(Xad, yS, XT))
            MT, MS = M[T].astype(float), M[S].astype(float); mask_in = ap_(yT, oof(MT, yT, folds)); mask_tr = ap_(yT, fit_pred(MS, yS, MT))
            rows.append(dict(db=a.db, site=a.site, n_T=int(T.sum()), pos_T=int(yT.sum()), prevalence_T=float(yT.mean()), regime_distance=dist, in_site=in_site, transfer=transfer, adapted=adapted, mask_only_in=mask_in, mask_only_transfer=mask_tr, gain_adapted=adapted - transfer, delta_transfer=transfer - in_site))
        print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__ == "__main__": main()
