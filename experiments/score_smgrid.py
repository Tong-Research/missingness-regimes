"""Score SMGRID against prereg/SMGRID.md. Prints every registered prediction and its verdict."""
import glob, pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results/smgrid"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "*.csv")))], ignore_index=True)
print(f"{d.rep.nunique()} replicates, {len(d)} rows\n")
clean = d.pivot_table(index=["gamma", "selfmask_rate"], columns="alignment", values="n_detect_clean")
gap = (clean["aligned"] - clean["closed"]).unstack("selfmask_rate")
print("n_detect_clean, aligned minus control (of six never-self-masking columns)")
print(gap.round(2).to_string(), "\n")
sm = d[d.alignment == "closed"].pivot_table(index="selfmask_rate", values="n_detect_sm", aggfunc="mean")
print("positive control: closed-mask detections on the six SELF-MASKED columns")
print(sm.round(2).to_string(), "\n")
rates = sorted(d.selfmask_rate.unique())
ok = {}
ok["S1 gamma=1 gap >= 2.0 at every s"] = bool((gap.loc[1.0] >= 2.0).all())
ok["S2 gamma=0 gap <= 1.0 at every s"] = bool((gap.loc[0.0].abs() <= 1.0).all())
ok["S3 mean gap larger at gamma=2 than gamma=1"] = bool(gap.loc[2.0].mean() > gap.loc[1.0].mean())
mono = sm.n_detect_sm.values
ok["S4 control detections rise with s and > 3.0 at s=0.60"] = bool(
    all(mono[i] <= mono[i + 1] + 1e-9 for i in range(len(mono) - 1)) and mono[-1] > 3.0)
ok["S5 gamma=1 gap at s=0.60 >= half its value at s=0"] = bool(
    gap.loc[1.0, rates[-1]] >= 0.5 * gap.loc[1.0, rates[0]])
for k, v in ok.items(): print(("  HELD    " if v else "  FAILED  ") + k)
print("\n" + ("ALL FIVE HELD" if all(ok.values()) else "NOT ALL HELD -- see prereg/SMGRID.md withdrawal conditions"))
sys.exit(0 if all(ok.values()) else 1)
