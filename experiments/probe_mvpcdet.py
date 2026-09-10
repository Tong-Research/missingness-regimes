"""MVPCDET (prereg/MVPCDET.md): the instrumented version of Result 215's P5.

Result 215 measured the misdiagnosis with a stand-in for MVPC's detection step. This script calls MVPC's own step,
`get_parent_missingness_pairs`, which is Step 1 of `mvpc_alg` in causal-learn and is invoked there exactly as it is
here, on the same plasmode substrate, the same replicate seeds and the same conditions as prereg/PLASMODE.md.

Under an aligned transplanted mask with no self-masking the transplanted mask is independent of the simulated
variables within a hospital by construction, so every parent the step reports is a false positive. Regime-affected
variables are the children of the latent regime and all their descendants in the generating graph: only these carry
the hospital-dependent mean shift, so only these can produce the spurious dependence."""
import argparse, csv, pathlib, sys, warnings, numpy as np
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from probe_plasmode import substrate, random_dag, simulate, spurious_parents, P, N, NCHILD, ALPHA
from causallearn.search.ConstraintBased.PC import get_parent_missingness_pairs
from causallearn.utils.cit import CIT

def descendants(A, seeds):
    """A[i, j] True means i -> j. Returns seeds plus everything reachable from them."""
    out = set(seeds); frontier = list(seeds)
    while frontier:
        i = frontier.pop()
        for j in np.flatnonzero(A[i]):
            if j not in out: out.add(int(j)); frontier.append(int(j))
    return out

def main():
    a = argparse.ArgumentParser(); a.add_argument("--rep", type=int, required=True); a.add_argument("--out", required=True); a = a.parse_args()
    Mall, gall = substrate(); rng = np.random.default_rng(1000 + a.rep)
    idx = rng.choice(len(gall), N, replace=False); mask0, zid = Mall[idx], gall[idx]
    A, order = random_dag(rng); ch = list(order[:NCHILD]); affected = descendants(A, ch); rows = []
    cells = [(g, al, False) for g in (0.0, 1.0, 2.0) for al in ("aligned", "closed")] + [(1.0, al, True) for al in ("aligned", "closed")]
    for gamma, align, sm in cells:
        r2 = np.random.default_rng(hash((a.rep, gamma, align, sm)) % 2**32)
        mask = mask0 if align == "aligned" else mask0[r2.permutation(len(mask0))]
        _, Xm, _ = simulate(np.random.default_rng(2000 + a.rep), A, order, gamma, zid, mask, sm)
        prt_m = get_parent_missingness_pairs(Xm, ALPHA, CIT(Xm, "mv_fisherz"), True)
        inds = list(prt_m["m"]); prts = [list(np.atleast_1d(p)) for p in prt_m["prt"]]
        flat = [int(q) for p in prts for q in p]
        rows.append(dict(rep=a.rep, gamma=gamma, alignment=align, selfmask=int(sm),
                         n_indicators_with_parent=len(inds), n_parent_pairs=len(flat),
                         share_affected=(float(np.mean([q in affected for q in flat])) if flat else np.nan),
                         share_child=(float(np.mean([q in ch for q in flat])) if flat else np.nan),
                         n_affected=len(affected), standin=spurious_parents(Xm, mask), miss_rate=float(np.isnan(Xm).mean())))
        print(f"  rep {a.rep} gamma {gamma} {align} selfmask {int(sm)}: indicators-with-parent {len(inds)} pairs {len(flat)} "
              f"share-regime-affected {rows[-1]['share_affected']} standin {rows[-1]['standin']}", flush=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
if __name__ == "__main__": main()
