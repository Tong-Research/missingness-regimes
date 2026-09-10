"""Score prereg/RRMIRROR.md: the corrected double dissociation, confirmed on the mirror split."""
import glob, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; V = lambda ok: "HOLDS" if ok else "FAILS"
REG = ["cand_eicu_regime", "cand_mimic4_regime", "cand_nhanes", "sweep_42739"]
STR = ["cand_acs_income", "cand_airbnb", "cand_higgs", "sweep_42737", "sweep_46654", "sweep_46703"]
def tab(d):
    fs = sorted(glob.glob(str(ROOT / f"results/{d}/*.csv"))); x = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    m = x[x.kind == "auroc"].groupby(["dataset", "model"]).value.mean().unstack("model")
    lk = {int(c.split("_")[1]): c for c in m.columns if c.startswith("lca_")}
    m["lca_best20"] = m[[lk[k] for k in sorted(lk) if k <= 20]].max(axis=1); m["diff"] = m["values"] - m.panels_tree; return m
mi, dv = tab("rrmirror"), tab("rrstruct")
print(f"=== RRMIRROR ({len(mi)}/17 datasets, fit on holdout, evaluated on dev) ===")
d1 = mi.reindex(REG)["diff"]; d2 = mi.reindex(STR)["diff"]
print(f"D1 regime sets, values - panels <= -0.15 on {int((d1 <= -0.15).sum())}/4: {V(int((d1 <= -0.15).sum()) == 4)}   " + " ".join(f"{k.split('_')[-1]}:{v:+.3f}" for k, v in d1.items()))
print(f"D2 structural sets, values - panels >= +0.10 on {int((d2 >= 0.10).sum())}/6: {V(int((d2 >= 0.10).sum()) == 6)}   " + " ".join(f"{k.split('_')[-1]}:{v:+.3f}" for k, v in d2.items()))
agree = int((np.sign(mi["diff"]) == np.sign(dv.reindex(mi.index)["diff"])).sum()); print(f"D3 sign of (values - panels) agrees with the dev-fit run on {agree}/{len(mi)}: {V(agree >= 15)}")
r = mi.lca_best20 / mi.panels_tree; p4 = int((r >= 0.95).sum()); print(f"D4 best mixture K<=20 >= 0.95 x tree on {p4}/{len(mi)} (needs >= 12): {V(p4 >= 12)}")
out = pd.concat([mi[["values", "panels_tree", "diff"]].round(3), dv.reindex(mi.index)["diff"].rename("diff_devfit").round(3)], axis=1)
out["group"] = ["regime" if i in REG else "structural" if i in STR else "other" for i in out.index]
print(out.sort_values(["group", "diff"]).to_string())
