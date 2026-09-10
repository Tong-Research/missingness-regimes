"""Score prereg/PLASMODE.md from results/plasmode/*.csv."""
import glob, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; V = lambda ok: "HOLDS" if ok else "FAILS"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results/plasmode/*.csv")))], ignore_index=True)
m = d[d.selfmask == 0]
print(f"=== PLASMODE ({d.rep.nunique()} replicates) ===")
w = m.pivot_table(index=["gamma", "rep"], columns=["arm", "alignment"], values="excess")
gap = {a: (w[(a, "aligned")] - w[(a, "closed")]) for a in m.arm.unique() if a != "complete"}
g1 = gap["td_pc"].xs(1.0, level="gamma"); p1 = g1.mean() >= 1.0 and (g1 > 0).mean() >= 0.70
print(f"P1 td_pc aligned-minus-control at gamma 1: mean {g1.mean():+.2f} edges (needs >= 1.0), positive in {(g1>0).mean():.0%} (needs >= 70%): {V(p1)}")
by = pd.DataFrame({a: v.groupby("gamma").mean() for a, v in gap.items()})
mono = bool(np.all(np.diff(by.td_pc.reindex([0.0, 1.0, 2.0]).values) >= -1e-9)); z = by.td_pc.get(0.0, np.nan)
print(f"P2 td_pc gap by gamma: " + " ".join(f"{k}:{v:+.2f}" for k, v in by.td_pc.items()) + f" non-decreasing {mono}, gamma 0 gap {z:+.2f} (needs <= 0.5): {V(mono and abs(z) <= 0.5)}")
r3 = by.mvpc.get(1.0, np.nan) / by.td_pc.get(1.0, np.nan) if by.td_pc.get(1.0, 0) else np.nan
print(f"P3 mvpc gap {by.mvpc.get(1.0, np.nan):+.2f} vs td_pc {by.td_pc.get(1.0, np.nan):+.2f}, ratio {r3:.2f} (needs >= 0.7): {V(r3 >= 0.7)}")
r4 = by.cdnod_est.get(1.0, np.nan) / by.td_pc.get(1.0, np.nan) if by.td_pc.get(1.0, 0) else np.nan
d4 = abs(w[("cdnod_est", "aligned")].xs(1.0, level="gamma").mean() - w[("cdnod_true", "aligned")].xs(1.0, level="gamma").mean())
print(f"P4 cdnod_est gap ratio {r4:.2f} (needs <= 0.5) and within {d4:.2f} edges of cdnod_true (needs <= 1.0): {V(r4 <= 0.5 and d4 <= 1.0)}")
sp = m.groupby(["gamma", "alignment", "rep"]).spurious.first().groupby(["gamma", "alignment"]).apply(lambda s: (s >= 1).mean())
print(f"P5 share of replicates with >= 1 falsely flagged column: " + " ".join(f"g{k[0]}/{k[1][:4]}:{v:.0%}" for k, v in sp.items()))
print(f"   at gamma 1 aligned: {sp.get((1.0,'aligned'), np.nan):.0%} (needs >= 50%): {V(sp.get((1.0,'aligned'), 0) >= 0.5)}")
print("\nmean skeleton distance to the true graph (target A), by arm:")
print(m.pivot_table(index=["gamma", "alignment"], columns="arm", values="shd_a").round(2).to_string())
print("\nmean excess over the complete-data arm:")
print(m.pivot_table(index=["gamma", "alignment"], columns="arm", values="excess").round(2).to_string())
print("\nspurious detections (mean count of falsely flagged columns):")
print(m.groupby(["gamma", "alignment"]).spurious.mean().round(2).to_string())
s2 = d[d.selfmask == 1]
if len(s2): print("\nsecondary, with self-masking at gamma 1:"); print(s2.pivot_table(index="alignment", columns="arm", values="excess").round(2).to_string())
