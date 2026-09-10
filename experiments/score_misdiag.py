"""Score prereg/MISDIAG.md from results/misdiag/*.csv (fresh seeds 20-39, causal-learn 0.1.4.7)."""
import glob, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; V = lambda ok: "HOLDS" if ok else "FAILS"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results/misdiag/*.csv")))], ignore_index=True)
m = d[d.selfmask == 0]; g = m.groupby(["gamma", "alignment"]).n_indicators_with_parent.mean()
a1, c1 = g[(1.0, "aligned")], g[(1.0, "closed")]; a0, c0 = g[(0.0, "aligned")], g[(0.0, "closed")]
a2, c2 = g[(2.0, "aligned")], g[(2.0, "closed")]
print(f"=== MISDIAG ({d.rep.nunique()} fresh replicates, seeds {int(d.rep.min())}-{int(d.rep.max())}) ===")
print(f"M1 aligned at realistic strength: {a1:.2f} (needs >= 3.0): {V(a1 >= 3.0)}")
print(f"M2 control at realistic strength: {c1:.2f} (needs <= 1.5): {V(c1 <= 1.5)}")
print(f"M3 difference: {a1-c1:.2f} (needs >= 3.0): {V(a1-c1 >= 3.0)}")
print(f"M4 difference at twice strength {a2-c2:.2f} > at zero {a0-c0:.2f}: {V((a2-c2) > (a0-c0))}")
print(f"M5 aligned and control differ by <= 1.0 at zero strength: {abs(a0-c0):.2f}: {V(abs(a0-c0) <= 1.0)}")
print("\nmean indicators reported with a parent, out of twelve:")
print(m.groupby(["gamma", "alignment"]).agg(instrumented=("n_indicators_with_parent", "mean"), pairs=("n_parent_pairs", "mean"), standin=("standin", "mean")).round(2).to_string())
s = d[d.selfmask == 1]
if len(s): print("\nsecondary, with self-masking at realistic strength:"); print(s.groupby("alignment").agg(instrumented=("n_indicators_with_parent", "mean"), standin=("standin", "mean")).round(2).to_string())
