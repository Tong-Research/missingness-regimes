"""Score SMGRID2 against prereg/SMGRID2.md, on the pooled statistic the design has power for."""
import glob, pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results/smgrid2"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "*.csv")))], ignore_index=True)
w = d.pivot_table(index=["rep", "gamma", "selfmask_rate"], columns="alignment",
                  values="n_detect_clean").reset_index()
w["gap"] = w["aligned"] - w["closed"]
print(f"{w.rep.nunique()} replicates, seeds {int(w.rep.min())}-{int(w.rep.max())}\n")
cell = w.pivot_table(index="gamma", columns="selfmask_rate", values="gap")
print("n_detect_clean, aligned minus control (six never-self-masking columns)")
print(cell.round(2).to_string(), "\n")
pool = {}
for g in (0.0, 1.0, 2.0):
    s = w[w.gamma == g].gap
    m, e = s.mean(), s.sem(); pool[g] = (m, e, m - 1.96 * e, m + 1.96 * e)
    print(f"  gamma={g}: pooled {m:+.3f} +- {e:.3f}   95% CI [{m-1.96*e:+.2f}, {m+1.96*e:+.2f}]")
sm = d[d.alignment == "closed"].groupby("selfmask_rate").n_detect_sm.mean()
print(f"\npositive control (closed mask, six self-masked columns): " +
      " -> ".join(f"{v:.2f}" for v in sm.values))
b = np.polyfit(w[w.gamma == 1.0].selfmask_rate, w[w.gamma == 1.0].gap, 1)[0]
print(f"slope of gap on s at gamma=1: {b:+.3f} per unit\n")
ok = {
 "T1 gamma=1 pooled >= 1.5 and CI excludes 0": bool(pool[1.0][0] >= 1.5 and pool[1.0][2] > 0),
 "T2 gamma=0 pooled <= 0.75 and CI includes 0": bool(pool[0.0][0] <= 0.75 and pool[0.0][2] <= 0 <= pool[0.0][3]),
 "T3 gamma=2 pooled > gamma=1 pooled": bool(pool[2.0][0] > pool[1.0][0]),
 "T4 positive control rises by >= 3.0 from s=0 to s=0.60": bool(sm.iloc[-1] - sm.iloc[0] >= 3.0),
 "T5 gamma=1 gap at s=0.60 >= 1.2": bool(cell.loc[1.0, 0.60] >= 1.2),
 "T6 slope on s at gamma=1 within +/-1.5": bool(abs(b) <= 1.5),
}
for k, v in ok.items(): print(("  HELD    " if v else "  FAILED  ") + k)
print("\n" + ("ALL SIX HELD" if all(ok.values()) else "NOT ALL HELD -- see prereg/SMGRID2.md withdrawal"))
sys.exit(0 if all(ok.values()) else 1)
