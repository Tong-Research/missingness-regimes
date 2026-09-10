"""Score ZERODEL against prereg/ZERODEL.md."""
import glob, pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results/zerodel"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "*.csv")))], ignore_index=True)
print(f"{d.rep.nunique()} replicates x {int(d.n_triples.iloc[0])} triples, alpha {d.alpha.iloc[0]}\n")
for col, lab in (("agree_cond", "their rule: delete on the CONDITIONING set"),
                 ("agree_all", "their alternative: delete on ALL involved variables")):
    t = d.pivot_table(index="gamma", columns="alignment", values=col, aggfunc=["mean", "sem"])
    print(lab)
    print(t.round(4).to_string(), "\n")
w = d.pivot_table(index=["rep", "gamma"], columns="alignment", values="agree_cond").reset_index()
w["gap"] = w["closed"] - w["aligned"]          # control minus aligned: positive = their rule degrades
g = w.groupby("gamma").gap.agg(["mean", "sem"])
print("their rule, control minus aligned (positive means the regime degrades it)")
print(g.round(4).to_string(), "\n")
wa = d.pivot_table(index=["rep", "gamma"], columns="alignment", values="agree_all").reset_index()
wa["gap"] = wa["closed"] - wa["aligned"]
print("same for the delete-on-all alternative")
print(wa.groupby("gamma").gap.agg(["mean", "sem"]).round(4).to_string(), "\n")
cells = d.groupby(["gamma", "alignment"])[["agree_cond", "agree_all"]].mean()
ok = {
 "Z1 gap >= 0.03 at gamma=2": bool(g.loc[2.0, "mean"] >= 0.03),
 "Z2 gap larger at gamma=2 than gamma=0": bool(g.loc[2.0, "mean"] > g.loc[0.0, "mean"]),
 "Z3 gap within +/-0.02 at gamma=0 (null control)": bool(abs(g.loc[0.0, "mean"]) <= 0.02),
 "Z4 delete-on-all >= their rule in every cell": bool((cells.agree_all >= cells.agree_cond).all()),
}
for k, v in ok.items(): print(("  HELD    " if v else "  FAILED  ") + k)
print("\n" + ("ALL FOUR HELD" if all(ok.values()) else "NOT ALL HELD -- see prereg/ZERODEL.md withdrawal"))
sys.exit(0)
