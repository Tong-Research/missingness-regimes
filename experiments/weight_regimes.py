"""Which merge weight wins, at each privacy budget, in each domain?

Result 27 established three regimes on clinical sites: with no privacy the best own-weight
is `w = min(1, EPV/10)`; under differential privacy it is n-squared; and FedAvg's plain
n-weighting is optimal in neither. That is the most surprising claim in the file and it
was fitted entirely on hospitals.

No new compute is needed to test it out of domain. `dp_merge.py` already evaluates all
four weightings in every cell it runs -- uniform, n, n^2, and the EPV-derived blend -- so
the ACS runs on disk contain the answer and have simply never been read for it. This
reads them.

**Prediction, recorded before running.** If the n^2 result is a fact about how DP noise
interacts with unequal site sizes, it should reappear on Census states, where the
amplification is largest (Result 36 measured 12.6x at d=10). If instead n^2 wins only on
clinical sites, it is a fact about the eICU size distribution and Result 27 must be
narrowed to it.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

ARMS = {"benefit": "n^1 (FedAvg)", "benefit_n2": "n^2",
        "benefit_unif": "uniform", "benefit_wepv": "w=min(1,EPV/10)",
        "benefit_w5": "w=0.5 fixed"}


def mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var / len(v))          # standard error, not sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="dp_*.csv produced by dp_merge.py")
    a = ap.parse_args()

    for path in a.files:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        arms = [k for k in ARMS if k in rows[0]]
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            eps = r["eps"]
            for k in arms:
                try:
                    by[eps][k].append(float(r[k]))
                except (ValueError, KeyError):
                    pass

        print(f"\n=== {path} — {len(rows)} decisions ===")
        head = f"{'eps':>6s} " + " ".join(f"{ARMS[k]:>16s}" for k in arms) + "   winner"
        print(head)
        for eps in sorted(by, key=lambda e: (e == "inf", float(e) if e != "inf" else 0)):
            cells, best, bestv = [], None, -9e9
            for k in arms:
                m, se = mean_sd(by[eps][k])
                cells.append(f"{m:+11.4f}±{se:.4f}" if se == se else f"{m:+16.4f}")
                if m > bestv:
                    best, bestv = k, m
            print(f"{eps:>6s} " + " ".join(f"{c:>16s}" for c in cells)
                  + f"   {ARMS[best]}")

        # Does the winner change with the privacy budget? That transition is the claim.
        winners = {}
        for eps in by:
            winners[eps] = max(arms, key=lambda k: mean_sd(by[eps][k])[0])
        distinct = set(winners.values())
        print(f"  winners across budgets: {sorted(distinct)}")
        if len(distinct) == 1:
            print("  -> no regime change here: one weighting wins at every budget.")
        else:
            print("  -> regime change present: the optimal weight depends on epsilon.")


if __name__ == "__main__":
    main()
