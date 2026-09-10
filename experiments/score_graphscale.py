"""Score GRAPHSCALE against prereg/GRAPHSCALE.md."""
import glob, pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results/graphscale"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "*.csv")))], ignore_index=True)
d["cfg"] = "p" + d.p.astype(str) + " d" + d.density.astype(str)
print(f"{len(d)} rows, configurations: {sorted(d.cfg.unique())}")
print(f"replicates per configuration: {d.groupby('cfg').rep.nunique().to_dict()}\n")
w = d.pivot_table(index=["cfg", "p", "density", "gamma", "arm"], columns="alignment",
                  values="excess").reset_index()
w["diff"] = w["aligned"] - w["closed"]
tab = w.pivot_table(index=["cfg", "arm"], columns="gamma", values="diff")
print("excess error, aligned minus control (edges); a null sits at 0")
print(tab.round(2).to_string(), "\n")
comp = d.groupby(["cfg", "p"]).shd_complete.mean().reset_index().sort_values("p")
print("complete-data arm's own skeleton error (is the problem actually harder?)")
print(comp.to_string(index=False), "\n")
ok = {}
g1 = w[(w.p == 12) & (w.density == 1.5) & (w.gamma == 1.0)]["diff"]
ok["G1 p=12 gamma=1 within +/-1.0 for every arm"] = bool((g1.abs() <= 1.0).all())
big = w[(w.density == 1.5) & (w.p.isin([20, 30]))]["diff"]
ok["G2 p=20 and p=30 within +/-2.0 for every arm and gamma"] = bool((big.abs() <= 2.0).all())
mono = []
for arm in w.arm.unique():
    v = [w[(w.arm == arm) & (w.p == p) & (w.density == 1.5)]["diff"].mean() for p in (12, 20, 30)]
    mono.append(v[0] < v[1] < v[2] or v[0] > v[1] > v[2])
ok["G3 no arm grows monotonically with p"] = bool(not any(mono))
d25 = w[(w.p == 20) & (w.density == 2.5)]["diff"]
ok["G4 p=20 density 2.5 within +/-2.0"] = bool(len(d25) > 0 and (d25.abs() <= 2.0).all())
# G5 is about the p ladder AT FIXED DENSITY. An earlier version of this scorer pooled the
# density-2.5 configuration into the p=20 cell, which averaged 6.07 with 32.30 and made G5 look
# failed. That was a bug in the scorer written after the registration, not a failed prediction.
c = d[d.density == 1.5].groupby("p").shd_complete.mean()
ok["G5 complete-data error grows with p (at density 1.5)"] = bool(c.loc[12] < c.loc[20] < c.loc[30])
print(f"  p ladder at density 1.5: " + " < ".join(f"p{int(k)} {v:.2f}" for k, v in c.items()))
dd = d[d.p == 20].groupby("density").shd_complete.mean()
print(f"  density axis at p=20   : " + ", ".join(f"d{k} {v:.2f}" for k, v in dd.items()) + "\n")
for k, v in ok.items(): print(("  HELD    " if v else "  FAILED  ") + k)
print("\n" + ("ALL FIVE HELD" if all(ok.values()) else "NOT ALL HELD -- see prereg/GRAPHSCALE.md withdrawal"))
sys.exit(0 if all(ok.values()) else 1)
