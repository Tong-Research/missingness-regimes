"""UNSEEN (prereg/UNSEEN.md): how much of a fresh sample lands on a missingness pattern never seen in training?
Good-Turing P0 = (singleton patterns) / N on the fitting half, Chao1 richness, the Heaps exponent (log-log slope of
distinct patterns vs rows on the fitting half, last decade), and the empirical share of the other half's rows whose
pattern is absent from (or has support < 30 in) the fitting half. Both directions (dev->holdout, holdout->dev)."""
import argparse, csv, pathlib, numpy as np
from probe_common import keys
import probe_common  # noqa: F401  (path setup)
from screen_candidates import dev_mask
from mimic_split import dev_mask as mimic_dev
C = pathlib.Path.home() / ".cache/phd-matrices"

def load_full(name):
    """Full matrix, NO per-slice column drop: run_hcal.load drops all-empty columns per half, which shifts column indices
    between halves and made every holdout pattern look unseen in the smoke test (2026-09-06 11:35)."""
    z = np.load(C / ("mimic4_sites.npz" if name == "mimic4" else f"{name}.npz"), allow_pickle=True); X = np.asarray(z["X"], float)
    m = mimic_dev(len(X)) if name == "mimic4" else dev_mask(name, len(X)); return X[m], X[~m]
from collections import Counter

def one(Xf, Xo, direction, ds):
    kf, ko = keys(~np.isnan(Xf)), keys(~np.isnan(Xo)); c = Counter(kf); N = len(kf); n1 = sum(1 for v in c.values() if v == 1); n2 = sum(1 for v in c.values() if v == 2)
    S = len(c); chao1 = S + (n1 * n1 / (2 * n2) if n2 else n1 * (n1 - 1) / 2)
    unseen = np.mean([e not in c for e in ko]); thin = np.mean([c.get(e, 0) < 30 for e in ko])
    rng = np.random.default_rng(0); perm = rng.permutation(N); ns = np.unique(np.geomspace(100, N, 12).astype(int)); S_n = np.array([len(set(kf[i] for i in perm[:m])) for m in ns])
    lo = ns >= N / 10; heaps = np.polyfit(np.log(ns[lo]), np.log(S_n[lo]), 1)[0] if lo.sum() >= 3 else np.nan
    return dict(dataset=ds, direction=direction, N_fit=N, N_other=len(ko), patterns_fit=S, singletons=n1, doubletons=n2, gt_p0=n1 / N, chao1=chao1, heaps=heaps, unseen_share=unseen, thin_share=thin)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    Xd, Xh = load_full(a.dataset)
    rows = [one(Xd, Xh, "dev->holdout", a.dataset), one(Xh, Xd, "holdout->dev", a.dataset)]
    for r in rows: print(f"{a.dataset} {r['direction']}: N={r['N_fit']:,} patterns={r['patterns_fit']:,} GT p0={r['gt_p0']:.4f} unseen={r['unseen_share']:.4f} thin={r['thin_share']:.4f} heaps={r['heaps']:.3f} chao1={r['chao1']:.0f}", flush=True)
    with open(a.out, "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__ == "__main__": main()
