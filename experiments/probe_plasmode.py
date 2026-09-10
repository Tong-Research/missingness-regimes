"""PLASMODE (prereg/PLASMODE.md): does a latent institutional regime that drives the missingness cost causal
discovery anything, over and above what it costs when it is merely a hidden confounder?

Plasmode simulation with a transplanted missingness mask (Gadbury 2008 for the term; Gentry 2023 for the transplant).
Substrate: eICU rows from hospitals with >= 500 stays, and the real missingness of twelve fixed columns.
Truth: a random DAG over p = 12 variables, plus a latent root Z (the hospital) with four children. Variables are
linear Gaussian; a variable that is a child of Z has its mean shifted by gamma * 0.184 * delta[z, j], where 0.184 is
the measured median between-hospital standard deviation of standardised column means in real eICU, and delta is drawn
once per replicate. The mask is then transplanted from the real record of the same row.

ALIGNED keeps each row's own mask. The CLOSED-MISSINGNESS CONTROL (Holovchak et al. 2025) permutes mask rows across
the whole sample, which preserves the multiset of mask rows exactly, and therefore every marginal rate, every
indicator-indicator dependence and every per-test deletion sample size, while removing the path between the mask and
the variables.

Scored on skeletons: target A is the true DAG over X, which every arm should recover and none can fully, and the
headline is each arm's excess over the complete-data arm; target B is the latent-projection skeleton (target A plus a
clique on the children of Z), which is what a latent-aware method should recover."""
import argparse, csv, itertools, pathlib, sys, warnings, numpy as np
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
from scipy import stats
C = pathlib.Path.home() / ".cache/phd-matrices"
COLS = ["lab_24", "lab_17", "vent_FiO2", "vent_LPM_O2", "lab_31", "lab_32", "lab_29", "lab_28", "lab_30", "lab_22", "lab_0", "lab_18"]
P = len(COLS); N = 20_000; EDGES = 18; NCHILD = 4; ANCHOR = 0.184; ALPHA = 0.01; MINSITE = 500

def substrate():
    zr = np.load(C / "cand_eicu_regime.npz", allow_pickle=True); zs = np.load(C / "eicu_sites.npz", allow_pickle=True)
    X = np.asarray(zr["X"], float); g = np.asarray(zs["g"]).ravel(); cols = list(zr["cols"])
    idx = [cols.index(c) for c in COLS]; keep = np.isin(g, [s for s in np.unique(g) if (g == s).sum() >= MINSITE])
    return np.isnan(X[keep][:, idx]), g[keep]

def random_dag(rng):
    order = rng.permutation(P); A = np.zeros((P, P), bool); pairs = [(order[i], order[j]) for i in range(P) for j in range(i + 1, P)]
    for k in rng.choice(len(pairs), EDGES, replace=False): A[pairs[k][0], pairs[k][1]] = True   # parent -> child
    return A, order

def simulate(rng, A, order, gamma, zid, mask, selfmask, sm_cols=None):
    """selfmask: False/0 for none, True for the original rate of 0.30 on every column, or a float
    rate. sm_cols restricts self-masking to those column indices (SMGRID); None means all columns.
    The defaults reproduce the published PLASMODE and MISDIAG behaviour exactly."""
    n = len(zid); W = np.where(A, rng.uniform(0.5, 1.5, (P, P)) * rng.choice([-1, 1], (P, P)), 0.0)
    ch = list(order[:NCHILD]); sites = np.unique(zid); delta = {s: rng.normal(size=P) for s in sites}
    shift = np.zeros((n, P))
    if gamma > 0:
        D = np.array([delta[s] for s in zid]); shift[:, ch] = gamma * ANCHOR * D[:, ch]
    X = np.zeros((n, P))
    for j in order: X[:, j] = X @ W[:, j] + rng.normal(size=n) + shift[:, j]
    M = mask.copy()
    rate = 0.30 if selfmask is True else float(selfmask or 0.0)
    cols = range(P) if sm_cols is None else sm_cols
    if rate > 0:
        for j in cols:
            if M[:, j].mean() > 0.01:
                extra = stats.norm.cdf((X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-9)) > 0.85
                M[:, j] = M[:, j] | (extra & (rng.random(n) < rate))
    Xm = X.copy(); Xm[M] = np.nan; return X, Xm, ch

def skel(G): A = (G != 0); return (A | A.T)[:P, :P] & ~np.eye(P, dtype=bool)
def shd_skel(est, true_): return int(np.triu(est ^ true_, 1).sum())
def f1_skel(est, true_):
    tp = np.triu(est & true_, 1).sum(); fp = np.triu(est & ~true_, 1).sum(); fn = np.triu(~est & true_, 1).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))

def bmm(Pm, K, seed=0, iters=120, eps=1e-3):
    rng = np.random.default_rng(seed); n, B = Pm.shape; th = rng.uniform(0.25, 0.75, (K, B)); pi = np.full(K, 1 / K)
    for _ in range(iters):
        lg = Pm @ np.log(th.T + 1e-12) + (1 - Pm) @ np.log(1 - th.T + 1e-12) + np.log(pi + 1e-12)
        m = lg.max(1, keepdims=True); w = np.exp(lg - m); r = w / w.sum(1, keepdims=True)
        pi = r.mean(0) + 1e-12; pi /= pi.sum(); th = (r.T @ Pm + eps) / (r.sum(0)[:, None] + 2 * eps)
    return (Pm @ np.log(th.T + 1e-12) + (1 - Pm) @ np.log(1 - th.T + 1e-12) + np.log(pi + 1e-12)).argmax(1)

def spurious_parents(Xm, M):
    """MVPC's premise: missingness explained by observed values. Under a transplanted mask with no self-masking any
    detection is a false positive, because within a hospital the mask is independent of the simulated variables."""
    k = 0
    for j in range(P):
        if not (0.01 <= M[:, j].mean() <= 0.99): continue
        hit = False
        for q in range(P):
            if q == j: continue
            o = ~np.isnan(Xm[:, q])
            if o.sum() < 200: continue
            a, b = Xm[o, q][M[o, j]], Xm[o, q][~M[o, j]]
            if len(a) > 50 and len(b) > 50 and stats.ttest_ind(a, b, equal_var=False).pvalue < ALPHA / P: hit = True; break
        k += int(hit)
    return k

def run_arms(X, Xm, zid, mask):
    out = {}
    out["complete"] = skel(pc(X, alpha=ALPHA, indep_test="fisherz", show_progress=False).G.graph)
    out["td_pc"] = skel(pc(Xm, alpha=ALPHA, indep_test="mv_fisherz", show_progress=False).G.graph)
    out["mvpc"] = skel(pc(Xm, alpha=ALPHA, indep_test="mv_fisherz", mvpc=True, show_progress=False).G.graph)
    try: out["fci"] = skel(fci(Xm, independence_test_method="mv_fisherz", alpha=ALPHA, verbose=False, show_progress=False)[0].graph)
    except Exception: out["fci"] = np.zeros((P, P), bool)
    ztrue = np.asarray(zid, float).reshape(-1, 1)
    zest = bmm(( ~mask).astype(float), 10).astype(float).reshape(-1, 1)
    for nm, cx in (("cdnod_true", ztrue), ("cdnod_est", zest)):
        try: out[nm] = skel(cdnod(Xm, cx, ALPHA, "mv_fisherz", show_progress=False).G.graph)
        except Exception: out[nm] = np.zeros((P, P), bool)
    return out

def main():
    a = argparse.ArgumentParser(); a.add_argument("--rep", type=int, required=True); a.add_argument("--out", required=True); a = a.parse_args()
    Mall, gall = substrate(); rng = np.random.default_rng(1000 + a.rep)
    idx = rng.choice(len(gall), N, replace=False); mask0, zid = Mall[idx], gall[idx]
    A, order = random_dag(rng); true_a = ((A | A.T) & ~np.eye(P, dtype=bool))
    ch_holder = {}; rows = []
    cells = [(g, al, False) for g in (0.0, 1.0, 2.0) for al in ("aligned", "closed")] + [(1.0, al, True) for al in ("aligned", "closed")]
    for gamma, align, sm in cells:
        r2 = np.random.default_rng(hash((a.rep, gamma, align, sm)) % 2**32)
        mask = mask0 if align == "aligned" else mask0[r2.permutation(len(mask0))]
        X, Xm, ch = simulate(np.random.default_rng(2000 + a.rep), A, order, gamma, zid, mask, sm)
        true_b = true_a.copy()
        for i, j in itertools.combinations(ch, 2): true_b[i, j] = true_b[j, i] = True
        arms = run_arms(X, Xm, zid, mask); sp = spurious_parents(Xm, mask); base = shd_skel(arms["complete"], true_a)
        for nm, S in arms.items():
            rows.append(dict(rep=a.rep, gamma=gamma, alignment=align, selfmask=int(sm), arm=nm,
                             shd_a=shd_skel(S, true_a), f1_a=f1_skel(S, true_a), shd_b=shd_skel(S, true_b),
                             excess=shd_skel(S, true_a) - base, spurious=sp, n_true_edges=int(np.triu(true_a, 1).sum()),
                             miss_rate=float(np.isnan(Xm).mean())))
        print(f"  rep {a.rep} gamma {gamma} {align} selfmask {int(sm)}: " + " ".join(f"{k}={shd_skel(v, true_a)}" for k, v in arms.items()) + f" spurious={sp}", flush=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
if __name__ == "__main__": main()
