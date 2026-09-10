"""GRAPHSCALE (prereg/GRAPHSCALE.md): does Paper B's negative result survive larger and denser graphs?

Paper B reports that a latent institutional regime costs constraint-based structure learning nothing,
across five algorithms and three regime strengths. It then states the limitation itself: "The
simulation uses one substrate, twelve variables and one graph size; a cost that appears only in
larger or denser problems would not be visible here."

A null that has only been measured at one problem size is a weak null. This runs the same design at
p in {12, 20, 30} and at two edge densities, on the same eICU substrate and the same transplanted
mask, and reports the same statistic: the excess skeleton error over the complete-data arm, aligned
minus the closed-missingness control.

Everything not being varied is held at the published values: N = 20,000 rows, four children of the
latent regime, the regime anchored at 0.184, alpha = 0.01. The p columns are the p columns of the
eICU regime matrix with missing rates nearest the median of the band [0.05, 0.95], chosen by a rule
fixed before any outcome was read, and the published twelve are a subset of that band.
"""
import argparse, csv, pathlib, sys, warnings, numpy as np
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
C = pathlib.Path.home() / ".cache/phd-matrices"
N = 20_000; NCHILD = 4; ANCHOR = 0.184; ALPHA = 0.01; MINSITE = 500

def substrate_p(p):
    """The p usable columns closest to the median missing rate, so a larger graph is not bought by
    adding columns that are nearly always present or nearly always absent."""
    zr = np.load(C / "cand_eicu_regime.npz", allow_pickle=True)
    zs = np.load(C / "eicu_sites.npz", allow_pickle=True)
    X = np.asarray(zr["X"], float); g = np.asarray(zs["g"]).ravel(); cols = [str(c) for c in zr["cols"]]
    keep = np.isin(g, [s for s in np.unique(g) if (g == s).sum() >= MINSITE])
    M = np.isnan(X[keep]); rate = M.mean(0)
    band = [j for j in range(len(cols)) if 0.05 <= rate[j] <= 0.95]
    med = np.median(rate[band])
    idx = sorted(sorted(band, key=lambda j: abs(rate[j] - med))[:p])
    return M[:, idx], g[keep], [cols[j] for j in idx]

def random_dag(rng, p, edges):
    order = rng.permutation(p); A = np.zeros((p, p), bool)
    pairs = [(order[i], order[j]) for i in range(p) for j in range(i + 1, p)]
    for k in rng.choice(len(pairs), min(edges, len(pairs)), replace=False):
        A[pairs[k][0], pairs[k][1]] = True
    return A, order

def simulate(rng, A, order, gamma, zid, mask, p):
    n = len(zid)
    W = np.where(A, rng.uniform(0.5, 1.5, (p, p)) * rng.choice([-1, 1], (p, p)), 0.0)
    ch = list(order[:NCHILD]); delta = {s: rng.normal(size=p) for s in np.unique(zid)}
    shift = np.zeros((n, p))
    if gamma > 0:
        D = np.array([delta[s] for s in zid]); shift[:, ch] = gamma * ANCHOR * D[:, ch]
    X = np.zeros((n, p))
    for j in order: X[:, j] = X @ W[:, j] + rng.normal(size=n) + shift[:, j]
    Xm = X.copy(); Xm[mask] = np.nan
    return X, Xm

def skel(G, p): A = (G != 0); return (A | A.T)[:p, :p] & ~np.eye(p, dtype=bool)
def shd(est, true_): return int(np.triu(est ^ true_, 1).sum())

def bmm(Pm, K, seed=0, iters=120, eps=1e-3):
    rng = np.random.default_rng(seed); n, B = Pm.shape
    th = rng.uniform(0.25, 0.75, (K, B)); pi = np.full(K, 1 / K)
    for _ in range(iters):
        lg = Pm @ np.log(th.T + 1e-12) + (1 - Pm) @ np.log(1 - th.T + 1e-12) + np.log(pi + 1e-12)
        m = lg.max(1, keepdims=True); w = np.exp(lg - m); r = w / w.sum(1, keepdims=True)
        pi = r.mean(0) + 1e-12; pi /= pi.sum(); th = (r.T @ Pm + eps) / (r.sum(0)[:, None] + 2 * eps)
    return (Pm @ np.log(th.T + 1e-12) + (1 - Pm) @ np.log(1 - th.T + 1e-12) + np.log(pi + 1e-12)).argmax(1)

def run_arms(X, Xm, zid, mask, p):
    out = {"complete": skel(pc(X, alpha=ALPHA, indep_test="fisherz", show_progress=False).G.graph, p)}
    out["td_pc"] = skel(pc(Xm, alpha=ALPHA, indep_test="mv_fisherz", show_progress=False).G.graph, p)
    out["mvpc"] = skel(pc(Xm, alpha=ALPHA, indep_test="mv_fisherz", mvpc=True, show_progress=False).G.graph, p)
    try: out["fci"] = skel(fci(Xm, independence_test_method="mv_fisherz", alpha=ALPHA, verbose=False, show_progress=False)[0].graph, p)
    except Exception: out["fci"] = np.zeros((p, p), bool)
    zt = np.asarray(zid, float).reshape(-1, 1)
    ze = bmm((~mask).astype(float), 10).astype(float).reshape(-1, 1)
    for nm, cx in (("cdnod_true", zt), ("cdnod_est", ze)):
        try: out[nm] = skel(cdnod(Xm, cx, ALPHA, "mv_fisherz", show_progress=False).G.graph, p)
        except Exception: out[nm] = np.zeros((p, p), bool)
    return out

def main():
    a = argparse.ArgumentParser(); a.add_argument("--rep", type=int, required=True)
    a.add_argument("--p", type=int, required=True); a.add_argument("--density", type=float, default=1.5,
                   help="edges per node; 1.5 reproduces the published 18 edges at p=12")
    a.add_argument("--out", required=True); a = a.parse_args()
    p = a.p; edges = int(round(a.density * p))
    Mall, gall, cols = substrate_p(p)
    rng = np.random.default_rng(1000 + a.rep)
    idx = rng.choice(len(gall), N, replace=False); mask0, zid = Mall[idx], gall[idx]
    A, order = random_dag(rng, p, edges); true_a = ((A | A.T) & ~np.eye(p, dtype=bool))
    print(f"rep {a.rep} p={p} edges={edges} density={a.density} cols={cols[:4]}...", flush=True)
    rows = []
    for gamma in (0.0, 1.0, 2.0):
        for align in ("aligned", "closed"):
            r2 = np.random.default_rng(hash((a.rep, gamma, align, p, edges)) % 2**32)
            mask = mask0 if align == "aligned" else mask0[r2.permutation(len(mask0))]
            X, Xm = simulate(np.random.default_rng(2000 + a.rep), A, order, gamma, zid, mask, p)
            arms = run_arms(X, Xm, zid, mask, p)
            base = shd(arms["complete"], true_a)
            for nm, G in arms.items():
                if nm == "complete": continue
                rows.append(dict(rep=a.rep, p=p, edges=edges, density=a.density, gamma=gamma,
                                 alignment=align, arm=nm, shd=shd(G, true_a), shd_complete=base,
                                 excess=shd(G, true_a) - base, n_true_edges=int(true_a.sum() // 2),
                                 miss_rate=float(np.isnan(Xm).mean())))
            print(f"  gamma {gamma} {align:7s}: complete {base:3d} | " +
                  " ".join(f"{nm} {shd(G, true_a) - base:+d}" for nm, G in arms.items() if nm != "complete"),
                  flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")

if __name__ == "__main__": main()
