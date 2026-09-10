"""Score MISCOST against prereg/MISCOST.md, on the per-pair signed error averaged over replicates."""
import glob, pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results/miscost"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "*.csv")))], ignore_index=True)
nrep = d.rep.nunique()
print(f"{nrep} replicates, {len(d):,} pair-observations\n")
# per-pair mean signed error: noise averages down, systematic distortion does not
pp = d.groupby(["estimator", "alignment", "gamma", "pair"]).signed_err.mean().reset_index()
summ = pp.groupby(["estimator", "alignment", "gamma"]).signed_err.agg(
    mean="mean", rms=lambda x: float(np.sqrt(np.mean(x ** 2)))).reset_index()
for est in ("available", "mi_obs", "mi_class", "mi_site"):
    t = summ[summ.estimator == est].pivot(index="gamma", columns="alignment", values=["mean", "rms"])
    print(f"{est}:"); print(t.round(5).to_string()); print()
av = summ[summ.estimator == "available"].set_index(["alignment", "gamma"])
res = 0.0124 / np.sqrt(nrep)
print(f"resolvable bias on {nrep} replicates: {res:.4f}\n")
ok = {
 "C1 available mean signed error within +/-0.005, aligned, every gamma":
   bool(all(abs(av.loc[("aligned", g), "mean"]) <= 0.005 for g in (0.0, 1.0, 2.0))),
 "C2 aligned-minus-control within +/-0.005 at every gamma":
   bool(all(abs(av.loc[("aligned", g), "mean"] - av.loc[("closed", g), "mean"]) <= 0.005 for g in (0.0, 1.0, 2.0))),
 "C3 RMS at gamma=2 at most twice its value at gamma=0 (aligned)":
   bool(av.loc[("aligned", 2.0), "rms"] <= 2 * av.loc[("aligned", 0.0), "rms"]),
 "C4 no estimator's aligned-minus-control exceeds +/-0.01":
   bool(all(abs(summ[(summ.estimator == e) & (summ.alignment == "aligned") & (summ.gamma == g)]["mean"].iloc[0] -
                summ[(summ.estimator == e) & (summ.alignment == "closed") & (summ.gamma == g)]["mean"].iloc[0]) <= 0.01
            for e in summ.estimator.unique() for g in (0.0, 1.0, 2.0))),
}
for k, v in ok.items(): print(("  HELD    " if v else "  FAILED  ") + k)
print("\n" + ("ALL HELD" if all(ok.values()) else "NOT ALL HELD -- see prereg/MISCOST.md withdrawal"))
sys.exit(0 if all(ok.values()) else 1)
