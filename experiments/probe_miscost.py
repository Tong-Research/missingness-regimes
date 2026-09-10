"""MISCOST (prereg/MISCOST.md): does the missingness distort the quantity a discovery algorithm consumes?

The merged paper shows that an institutional regime makes MVPC's detection step report the observed
values as parents of the missingness indicators, when the dependence is induced by an unobserved
common cause. It is explicit that it does not measure what that costs. This measures it.

The estimand is the correlation matrix. That is not an arbitrary choice: a constraint-based
algorithm reads partial correlations off exactly this matrix, so a bias here is a bias in every
independence test the algorithm runs. Truth is the complete-data correlation on the same rows.

Under an aligned mask the rows on which a variable is observed are a hospital-biased subsample,
because the mask belongs to the hospital and the hospital shifts the variables. Pairwise deletion
therefore estimates the correlation on a distorted subsample. Under the closed-missingness control
the mask rows are permuted, so the same subsample is random and no bias should appear.

  available   pairwise-complete correlation -- what test-wise deletion uses
  mi_obs      correlation after imputation from the observed values alone
  mi_class    the same, imputed within each latent class estimated FROM THE MASK, no site info
  mi_site     the same, imputed within each true hospital -- the upper bound on knowing the regime

A NOTE ON THE FIRST DESIGN, which was written and discarded on 2026-09-07. It scored a stacked
vector of twelve leave-one-out regressions. With a 47% missing rate almost no row is complete on
twelve columns, so the available-case arm fell through its own guard and returned ZEROS -- and since
the true coefficients are mostly zero, returning zeros scored best of every arm. The lesson is that
an estimator which degrades to the null must never be scored by distance to a sparse truth. The
correlation matrix has no such failure mode: every pair has roughly 28% of rows observed on both.
"""
import argparse, csv, pathlib, sys, warnings, numpy as np
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from probe_plasmode import substrate, random_dag, simulate, bmm, P, N
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

KCLASS = 10
MINGRP = 300          # a group smaller than this keeps the pooled imputation
IU = np.triu_indices(P, 1)


def corr_complete(X):
    return np.corrcoef(X, rowvar=False)[IU]


def corr_pairwise(Xm):
    """Pairwise-complete correlation: each entry uses the rows where both variables are observed."""
    out = np.zeros(len(IU[0]))
    for k, (i, j) in enumerate(zip(*IU)):
        ok = ~np.isnan(Xm[:, i]) & ~np.isnan(Xm[:, j])
        out[k] = np.corrcoef(Xm[ok, i], Xm[ok, j])[0, 1] if ok.sum() >= 100 else np.nan
    return out


def impute(Xm, groups, seed):
    mk = lambda: IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=seed,
                                  keep_empty_features=True)
    F = mk().fit_transform(Xm)
    if groups is None: return F
    for g in np.unique(groups):
        at = np.flatnonzero(groups == g)
        if len(at) < MINGRP: continue
        try: F[at] = mk().fit_transform(Xm[at])
        except Exception: pass
    return F


def main():
    a = argparse.ArgumentParser(); a.add_argument("--rep", type=int, required=True)
    a.add_argument("--out", required=True); a = a.parse_args()
    Mall, gall = substrate(); rng = np.random.default_rng(1000 + a.rep)
    idx = rng.choice(len(gall), N, replace=False); mask0, zid = Mall[idx], gall[idx]
    A, order = random_dag(rng)
    rows = []
    for gamma in (0.0, 1.0, 2.0):
        for align in ("aligned", "closed"):
            r2 = np.random.default_rng(hash((a.rep, gamma, align)) % 2**32)
            mask = mask0 if align == "aligned" else mask0[r2.permutation(len(mask0))]
            X, Xm, _ = simulate(np.random.default_rng(2000 + a.rep), A, order, gamma, zid, mask, False)
            truth = corr_complete(X)
            cls = bmm((~mask).astype(float), KCLASS, seed=a.rep)
            est = {"available": corr_pairwise(Xm),
                   "mi_obs":   corr_complete(impute(Xm, None, a.rep)),
                   "mi_class": corr_complete(impute(Xm, cls, a.rep)),
                   "mi_site":  corr_complete(impute(Xm, np.asarray(zid), a.rep))}
            for nm, v in est.items():
                ok = np.isfinite(v) & np.isfinite(truth)
                # Per-PAIR signed error. Averaged across replicates the sampling noise cancels and
                # any systematic distortion does not, which is the only way this design can separate
                # "no bias" from "a bias below the noise floor of one sample".
                for k in np.flatnonzero(ok):
                    rows.append(dict(rep=a.rep, gamma=gamma, alignment=align, estimator=nm,
                                     pair=int(k), signed_err=float(v[k] - truth[k]),
                                     truth=float(truth[k]), miss_rate=float(np.isnan(Xm).mean())))
            print(f"  gamma {gamma} {align:7s}: " +
                  "  ".join(f"{k} rmse {np.sqrt(np.mean((v[np.isfinite(v) & np.isfinite(truth)] - truth[np.isfinite(v) & np.isfinite(truth)]) ** 2)):.4f}"
                            for k, v in est.items()), flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__": main()
