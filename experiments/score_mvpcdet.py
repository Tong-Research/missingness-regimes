"""Score prereg/MVPCDET.md from results/mvpcdet/*.csv."""
import glob, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; V = lambda ok: "HOLDS" if ok else "FAILS"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results/mvpcdet/*.csv")))], ignore_index=True)
m = d[d.selfmask == 0]
print(f"=== MVPCDET ({d.rep.nunique()} replicates) ===")
a1 = m[(m.gamma == 1.0) & (m.alignment == "aligned")]; c1 = m[(m.gamma == 1.0) & (m.alignment == "closed")]
q1 = (a1.n_indicators_with_parent >= 1).mean(); print(f"Q1 at least one indicator reported with a parent in {q1:.0%} of replicates (needs >= 90%): {V(q1 >= 0.90)}")
q2 = a1.n_indicators_with_parent.mean(); print(f"Q2 mean indicators with a parent, aligned: {q2:.2f} (needs >= 3.0): {V(q2 >= 3.0)}")
q3 = c1.n_indicators_with_parent.mean(); print(f"Q3 same under the closed-missingness control: {q3:.2f} (needs <= 0.5): {V(q3 <= 0.5)}")
by = m[m.alignment == "aligned"].groupby("gamma").n_indicators_with_parent.mean()
q4 = bool(np.all(np.diff(by.reindex([0.0, 1.0, 2.0]).values) >= -1e-9)); print(f"Q4 aligned mean by gamma: " + " ".join(f"{k}:{v:.2f}" for k, v in by.items()) + f" non-decreasing: {V(q4)}")
q5 = a1.share_affected.mean(); print(f"Q5 share of reported parents that are regime-affected: {q5:.3f} (needs >= 0.90): {V(q5 >= 0.90)}")
print(f"    (context: mean regime-affected variables per replicate {m.n_affected.mean():.1f} of 12, so the base rate is {m.n_affected.mean()/12:.2f})")
print("\ninstrumented step versus our stand-in, mean over replicates:")
print(m.groupby(["gamma", "alignment"]).agg(indicators_with_parent=("n_indicators_with_parent", "mean"), parent_pairs=("n_parent_pairs", "mean"), standin=("standin", "mean")).round(2).to_string())
s = d[d.selfmask == 1]
if len(s): print("\nsecondary, with self-masking at gamma 1:"); print(s.groupby("alignment").agg(indicators_with_parent=("n_indicators_with_parent", "mean"), standin=("standin", "mean")).round(2).to_string())
