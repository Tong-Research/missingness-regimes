"""SMGRID (prereg/SMGRID.md): separate the regime's contribution to false detections from a genuine
self-masking signal, by varying the two independently.

Paper B states the gap in its own words: "Separating the two would need a design in which the
strength of the self-masking and the strength of the regime are varied independently, which we did
not run." MISDIAG's secondary condition switched self-masking on for every column at one rate, so
once a real signal existed the step found it everywhere and the regime's share was not recoverable.

The fix is to give self-masking to only HALF the columns. Six columns censor themselves at rate s;
the other six never do. On those other six, under an aligned transplanted mask, every parent the
step reports is still a false positive at every value of s, because the transplanted mask is
independent of the simulated variables within a hospital by construction. So

    n_detect_clean   detections among the six columns that never self-mask   <- the isolated regime
    n_detect_sm      detections among the six that do                        <- regime plus signal

and the aligned-minus-control difference in `n_detect_clean` measures the regime alone, at any
self-masking strength. Grid: gamma in {0, 1, 2} x s in {0, 0.15, 0.30, 0.60} x {aligned, closed}.

The six self-masking columns are drawn once per replicate and recorded, so the split is not chosen
to suit the result.
"""
import argparse, csv, pathlib, sys, warnings, numpy as np
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from probe_plasmode import substrate, random_dag, simulate, P, N, NCHILD, ALPHA
from probe_mvpcdet import descendants
from causallearn.search.ConstraintBased.PC import get_parent_missingness_pairs
from causallearn.utils.cit import CIT

GAMMAS = (0.0, 1.0, 2.0)
RATES = (0.0, 0.15, 0.30, 0.60)

def main():
    a = argparse.ArgumentParser(); a.add_argument("--rep", type=int, required=True)
    a.add_argument("--out", required=True); a = a.parse_args()
    Mall, gall = substrate(); rng = np.random.default_rng(1000 + a.rep)
    idx = rng.choice(len(gall), N, replace=False); mask0, zid = Mall[idx], gall[idx]
    A, order = random_dag(rng); ch = list(order[:NCHILD]); affected = descendants(A, ch)

    # the six self-masking columns, drawn once per replicate, before any outcome is seen
    smc = sorted(np.random.default_rng(7000 + a.rep).choice(P, P // 2, replace=False).tolist())
    clean = [j for j in range(P) if j not in smc]
    print(f"rep {a.rep}: self-masking columns {smc}; clean columns {clean}", flush=True)

    rows = []
    for gamma in GAMMAS:
        for s in RATES:
            for align in ("aligned", "closed"):
                r2 = np.random.default_rng(hash((a.rep, gamma, s, align)) % 2**32)
                mask = mask0 if align == "aligned" else mask0[r2.permutation(len(mask0))]
                _, Xm, _ = simulate(np.random.default_rng(2000 + a.rep), A, order, gamma, zid,
                                    mask, s, sm_cols=smc)
                prt = get_parent_missingness_pairs(Xm, ALPHA, CIT(Xm, "mv_fisherz"), True)
                inds = [int(i) for i in np.atleast_1d(prt["m"])]
                flat = [int(q) for p in prt["prt"] for q in np.atleast_1d(p)]
                rows.append(dict(
                    rep=a.rep, gamma=gamma, selfmask_rate=s, alignment=align,
                    n_detect_total=len(inds),
                    n_detect_clean=sum(1 for i in inds if i in clean),
                    n_detect_sm=sum(1 for i in inds if i in smc),
                    n_clean_cols=len(clean), n_sm_cols=len(smc),
                    n_parent_pairs=len(flat),
                    share_affected=(float(np.mean([q in affected for q in flat])) if flat else np.nan),
                    miss_rate=float(np.isnan(Xm).mean()),
                    sm_cols="|".join(map(str, smc))))
                print(f"  gamma {gamma} s {s:.2f} {align:7s}: total {len(inds):2d} "
                      f"clean {rows[-1]['n_detect_clean']:2d}/{len(clean)} "
                      f"selfmasked {rows[-1]['n_detect_sm']:2d}/{len(smc)} "
                      f"missrate {rows[-1]['miss_rate']:.3f}", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")

if __name__ == "__main__": main()
