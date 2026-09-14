#!/usr/bin/env python3
"""CORRTRACE (prereg/CORRTRACE.md): reproduce the indicator-to-value correlation figures.

Two manuscripts print four numbers for this quantity with no macro behind them, and no result file
or script in the repository produces them. This recomputes them from the published simulation setup
so they can be confirmed, corrected, or withdrawn.

Setup is probe_mvpcdet.py's, unchanged: substrate() for the transplanted mask and hospital ids,
random_dag on default_rng(1000+rep), N rows without replacement, the ALIGNED mask, NO self-masking,
and simulate on default_rng(2000+rep).

One row per (rep, gamma, scope). Writes results/corrtrace/<rep>.csv.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from banked import BankedWriter                                   # noqa: E402
from probe_plasmode import N, P, substrate, random_dag, simulate  # noqa: E402

GAMMAS = (0.0, 1.0, 2.0)
FIELDS = ["rep", "gamma", "scope", "median_abs_corr", "n_pairs", "n_skipped_cols"]
SCOPES = ("pooled", "within_hospital")


def abs_corrs(M: np.ndarray, X: np.ndarray, rows: np.ndarray) -> tuple[list[float], int]:
    """|corr(indicator j, value k)| for every j != k, on the given rows.

    An indicator with no variation on these rows has an undefined correlation, not a zero one:
    counting it as zero would drag the median toward zero and manufacture the very smallness the
    sentence claims. Those columns are skipped and counted.
    """
    m, x = M[rows], X[rows]
    keep = m.std(axis=0) > 0
    skipped = int((~keep).sum())
    if keep.sum() == 0 or len(rows) < 3:
        return [], skipped
    mz = (m[:, keep] - m[:, keep].mean(0)) / m[:, keep].std(0)
    xs = x.std(0)
    xkeep = xs > 0
    if xkeep.sum() == 0:
        return [], skipped
    xz = (x[:, xkeep] - x[:, xkeep].mean(0)) / xs[xkeep]
    C = np.abs(mz.T @ xz) / len(rows)                     # |corr| for every kept (j, k)
    jj = np.flatnonzero(keep)[:, None]
    kk = np.flatnonzero(xkeep)[None, :]
    return C[jj != kk].tolist(), skipped


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--rep", type=int, required=True)
    a.add_argument("--out", required=True)
    a = a.parse_args()

    Mall, gall = substrate()
    rng = np.random.default_rng(1000 + a.rep)
    idx = rng.choice(len(gall), N, replace=False)
    mask0, zid = Mall[idx], gall[idx]
    A, order = random_dag(rng)
    sites = np.unique(zid)
    print(f"rep {a.rep}: n {N:,} p {P} hospitals {len(sites)}", flush=True)

    out = pathlib.Path(a.out)
    with BankedWriter(out, FIELDS, key=("gamma",), member="scope",
                      members=SCOPES, cast=float) as bank:
        done = bank.completed()
        for gamma in GAMMAS:
            if (gamma,) in done:
                continue
            X, Xm, _ = simulate(np.random.default_rng(2000 + a.rep), A, order,
                                gamma, zid, mask0, False)
            M = np.isnan(Xm)

            pooled, skip_p = abs_corrs(M, X, np.arange(N))
            within, skip_w = [], 0
            for s in sites:
                rows = np.flatnonzero(zid == s)
                cs, sk = abs_corrs(M, X, rows)
                within += cs
                skip_w += sk

            rows_out = []
            for scope, vals, sk in (("pooled", pooled, skip_p),
                                    ("within_hospital", within, skip_w)):
                med = float(np.median(vals)) if vals else float("nan")
                rows_out.append(dict(rep=a.rep, gamma=gamma, scope=scope,
                                     median_abs_corr=med, n_pairs=len(vals),
                                     n_skipped_cols=sk))
                print(f"  gamma {gamma}: {scope:<16} median |corr| {med:.5f} "
                      f"over {len(vals):,} pairs ({sk} column(s) skipped)", flush=True)
            bank.write_unit(rows_out)
        rc = bank.exit_code(len(GAMMAS))
    print(f"wrote {out} ({bank.banked} of {len(GAMMAS)} gamma cells)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
