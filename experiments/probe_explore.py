"""EXPLORE (prereg/EXPLORE.md): a fixed battery of missingness statistics, one dataset per job, on ONE slice (dev for the
hunt, holdout for confirmation). 3-fold stratified CV seed 0, HCAL tree config, cap 60,000 rows. Writes three CSVs:
<out>.dataset.csv (one row), <out>.columns.csv (one row per column), <out>.patterns.csv (one row per pattern with
support >= 100). Nothing here is a hypothesis; the battery is frozen before any output is read."""
import argparse, csv, numpy as np, pandas as pd
from probe_common import load, tree, keys, groups, SimpleImputer, StandardScaler, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as ap_, log_loss, roc_auc_score
from scipy import stats

def keepcols(F):
    """HistGB raises 'window shape cannot be larger than input array shape' on columns with < 2 distinct non-NaN values."""
    return [j for j in range(F.shape[1]) if len(np.unique(F[~np.isnan(F[:, j]), j])) >= 2] or [0]

def oof(F, y, folds, mk=tree):
    p = np.zeros(len(y))
    for tr, te in folds: kc = keepcols(F[tr]); p[te] = mk().fit(F[tr][:, kc], y[tr]).predict_proba(F[te][:, kc])[:, 1]
    return p

def main():
    a = argparse.ArgumentParser(); a.add_argument("--dataset", required=True); a.add_argument("--slice", default="dev"); a.add_argument("--out", required=True); a = a.parse_args()
    X, y = load(a.dataset, a.slice, 60_000); n, p = X.shape; M = np.isnan(X); S = (~M).sum(1); prev = y.mean(); rng = np.random.default_rng(0)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y)); ks = keys(~M); g = groups(ks)
    Xi = SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(X)
    P = {"native": oof(X, y, folds), "native_plus_mask": oof(np.hstack([X, M.astype(float)]), y, folds), "mask_only": oof(M.astype(float), y, folds), "support_only": oof(S[:, None].astype(float), y, folds)}
    P["lr_mean"] = np.zeros(n)
    for tr, te in folds:
        sc = StandardScaler().fit(Xi[tr]); P["lr_mean"][te] = LogisticRegressionCV(Cs=6, cv=3, max_iter=2000, scoring="average_precision").fit(sc.transform(Xi[tr]), y[tr]).predict_proba(sc.transform(Xi[te]))[:, 1]
    au = {k: ap_(y, v) for k, v in P.items()}
    # mask geometry
    c = np.unique(ks, return_counts=True)[1] if False else np.array([len(v) for v in g.values()]); ent = -np.sum((c / n) * np.log2(c / n))
    Mc = M.astype(float) - M.mean(0); sv = np.linalg.svd(Mc, compute_uv=False) if n * p < 5e7 else np.linalg.svd(Mc[rng.choice(n, 20000, replace=False)], compute_uv=False); cum = np.cumsum(sv ** 2) / np.sum(sv ** 2); rank90 = int(np.searchsorted(cum, 0.9) + 1) if np.sum(sv ** 2) > 0 else 0
    # outcome by pattern / support
    big = {e: r for e, r in g.items() if len(r) >= 30}; between = sum(len(r) * (y[r].mean() - prev) ** 2 for r in big.values()) / max(sum(len(r) for r in big.values()), 1) / max(prev * (1 - prev), 1e-9)
    sb = pd.qcut(S, min(10, len(np.unique(S))), duplicates="drop") if len(np.unique(S)) > 1 else pd.Series(np.zeros(n)); bs = pd.Series(y).groupby(np.asarray(sb.codes) if hasattr(sb, "codes") else np.zeros(n)).agg(["mean", "size"]); between_support = float(np.sum(bs["size"] * (bs["mean"] - prev) ** 2) / n / max(prev * (1 - prev), 1e-9))
    rho_sy = stats.spearmanr(S, y)[0] if len(np.unique(S)) > 1 else np.nan
    # per-pattern rows (support >= 100) on the native tree's oof predictions
    prow = []
    for e, r in g.items():
        if len(r) < 100: continue
        yy, pp = y[r], P["native"][r]; two = len(np.unique(yy)) == 2
        prow.append(dict(dataset=a.dataset, slice=a.slice, support=len(r), n_obs=len(e), prevalence=yy.mean(), auprc=ap_(yy, pp) if two else np.nan, auroc=roc_auc_score(yy, pp) if two else np.nan, logloss=log_loss(yy, pp, labels=[0, 1]), mean_pred=pp.mean(), calib_gap=pp.mean() - yy.mean(), lift=(ap_(yy, pp) / yy.mean()) if two and yy.mean() > 0 else np.nan))
    pr = pd.DataFrame(prow); ok = pr.dropna(subset=["auprc"]) if len(pr) else pr
    rho_sup_auprc = stats.spearmanr(ok.support, ok.auprc)[0] if len(ok) >= 5 else np.nan; rho_sup_lift = stats.spearmanr(ok.support, ok.lift)[0] if len(ok) >= 5 else np.nan; rho_nobs_prev = stats.spearmanr(pr.n_obs, pr.prevalence)[0] if len(pr) >= 5 else np.nan
    # column rows: rate, permutation importance (native tree, oof), recoverability, value shift with support
    crow = []; base = au["native"]; models = [(keepcols(X[tr]), tree().fit(X[tr][:, keepcols(X[tr])], y[tr])) for tr, _ in folds]
    for j in range(p):
        rate = M[:, j].mean(); Xp = X.copy(); Xp[:, j] = Xp[rng.permutation(n), j]; pp = np.zeros(n)
        for (tr, te), (kc, mdl) in zip(folds, models): pp[te] = mdl.predict_proba(Xp[te][:, kc])[:, 1]
        imp = base - ap_(y, pp); o = ~M[:, j]
        rec = rec1 = rec3 = np.nan; top = ""
        if 0.01 <= rate <= 0.99:
            Fo = np.delete(Xi, j, 1); t = M[:, j].astype(int); rec = roc_auc_score(t, oof(Fo, t, folds, lambda: tree(max_iter=150)))
            # policy sparsity: permutation importance of each other column for predicting M_j, then refit on the top 1 and top 3
            rm = [(keepcols(Fo[tr]), tree(max_iter=150).fit(Fo[tr][:, keepcols(Fo[tr])], t[tr])) for tr, _ in folds]; drops = []
            for q in range(Fo.shape[1]):
                Fq = Fo.copy(); Fq[:, q] = Fq[rng.permutation(n), q]; pq = np.zeros(n)
                for (tr, te), (kc, mdl) in zip(folds, rm): pq[te] = mdl.predict_proba(Fq[te][:, kc])[:, 1]
                drops.append(rec - roc_auc_score(t, pq))
            order = np.argsort(drops)[::-1]; others = [c for c in range(p) if c != j]; top = ";".join(str(others[q]) for q in order[:3])
            rec1 = roc_auc_score(t, oof(Fo[:, order[:1]], t, folds, lambda: tree(max_iter=150))); rec3 = roc_auc_score(t, oof(Fo[:, order[:3]], t, folds, lambda: tree(max_iter=150)))
        shift = stats.spearmanr(X[o, j], S[o])[0] if o.sum() > 50 and len(np.unique(X[o, j])) > 1 else np.nan
        cy = stats.spearmanr(X[o, j], y[o])[0] if o.sum() > 50 and len(np.unique(X[o, j])) > 1 and len(np.unique(y[o])) == 2 else np.nan
        crow.append(dict(dataset=a.dataset, slice=a.slice, col=j, rate=rate, importance=imp, recoverability=rec, recov_top1=rec1, recov_top3=rec3, top_predictors=top, shift_with_support=shift, corr_y=cy))
    cr = pd.DataFrame(crow); cc = cr[(cr.rate > 0.01) & (cr.rate < 0.99)]
    rho_rate_imp = stats.spearmanr(cc.rate, cc.importance)[0] if len(cc) >= 5 else np.nan; rho_rate_absy = stats.spearmanr(cc.rate, cc.corr_y.abs())[0] if cc.corr_y.notna().sum() >= 5 else np.nan
    # panel structure of the mask: columns whose indicators coincide (exact) or nearly (Jaccard >= 0.9 on the missing sets)
    mc = [j for j in range(p) if 0.01 <= M[:, j].mean() <= 0.99]; n_exact = len({M[:, j].tobytes() for j in mc}); groups_j = []
    for j in mc:
        for grp in groups_j:
            k = grp[0]; inter = (M[:, j] & M[:, k]).sum(); uni = (M[:, j] | M[:, k]).sum()
            if uni and inter / uni >= 0.9: grp.append(j); break
        else: groups_j.append([j])
    row = dict(dataset=a.dataset, slice=a.slice, n=n, p=p, n_panels_exact=n_exact, n_panels_j90=len(groups_j), median_recov_top1=cr.recov_top1.median() if len(cr) else np.nan, median_recov_top3=cr.recov_top3.median() if len(cr) else np.nan, share_recov_top3_ge95=float((cr.recov_top3 >= 0.95).mean()) if cr.recov_top3.notna().any() else np.nan, prevalence=prev, n_patterns=len(g), singleton_share=float((c == 1).sum() / n), mean_support_frac=S.mean() / p, mask_entropy_bits=ent, mask_rank90=rank90, cols_with_missing=int(((M.mean(0) > 0.01) & (M.mean(0) < 0.99)).sum()),
               **{f"auprc_{k}": v for k, v in au.items()}, mask_gain=au["native_plus_mask"] - au["native"], mask_only_lift=au["mask_only"] / prev, support_only_lift=au["support_only"] / prev, tree_minus_lr=au["native"] - au["lr_mean"],
               between_pattern_var_share=between, between_support_var_share=between_support, rho_support_y=rho_sy, n_patterns_ge100=len(pr), rho_support_auprc=rho_sup_auprc, rho_support_lift=rho_sup_lift, rho_nobs_prevalence=rho_nobs_prev,
               pattern_auprc_range=(ok.auprc.max() - ok.auprc.min()) if len(ok) else np.nan, pattern_calib_maxabs=pr.calib_gap.abs().max() if len(pr) else np.nan, rho_rate_importance=rho_rate_imp, rho_rate_abscorr_y=rho_rate_absy, median_recoverability=cc.recoverability.median() if len(cc) else np.nan, median_abs_shift=cc.shift_with_support.abs().median() if len(cc) else np.nan)
    pd.DataFrame([row]).to_csv(a.out + ".dataset.csv", index=False); cr.to_csv(a.out + ".columns.csv", index=False); pr.to_csv(a.out + ".patterns.csv", index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
if __name__ == "__main__": main()
