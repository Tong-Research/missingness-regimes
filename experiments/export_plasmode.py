"""Write one replicate of the PLASMODE design to an npz, so a third-party method can be run on
EXACTLY the data our own arms see, in its own environment. Nothing about the design changes here."""
import argparse, json, pathlib, sys, warnings, numpy as np
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from probe_plasmode import substrate, random_dag, simulate, P, N

def main():
    a = argparse.ArgumentParser(); a.add_argument("--rep", type=int, required=True)
    a.add_argument("--out", required=True); a = a.parse_args()
    Mall, gall = substrate(); rng = np.random.default_rng(1000 + a.rep)
    idx = rng.choice(len(gall), N, replace=False); mask0, zid = Mall[idx], gall[idx]
    A, order = random_dag(rng)
    out, cells = {}, []
    for gamma in (0.0, 1.0, 2.0):
        for align in ("aligned", "closed"):
            r2 = np.random.default_rng(hash((a.rep, gamma, align)) % 2**32)
            mask = mask0 if align == "aligned" else mask0[r2.permutation(len(mask0))]
            X, Xm, _ = simulate(np.random.default_rng(2000 + a.rep), A, order, gamma, zid, mask, False)
            key = f"g{gamma:.0f}_{align}"
            out[f"{key}_X"], out[f"{key}_Xm"] = X.astype(np.float32), Xm.astype(np.float32)
            cells.append(dict(key=key, gamma=gamma, alignment=align))
    out["meta"] = json.dumps(dict(rep=a.rep, p=P, n=N, cells=cells,
                                  true_adj=A.astype(int).tolist(), order=order.tolist()))
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: {len(cells)} cells, p={P}, n={N}")

if __name__ == "__main__": main()
