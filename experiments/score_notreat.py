"""Score NOTREAT against prereg/NOTREAT.md."""
import pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results/notreat"
ok = {}
for tag in ("eicu", "mimic4", "eicu-mirror"):
    f = R / f"{tag}.csv"
    if not f.exists(): print(f"{tag}: MISSING"); continue
    d = pd.read_csv(f); a = d[d.kind == "auroc"]
    w = a.pivot_table(index="panel", columns="model", values="value")
    pt, vl = w["panels_tree"], w.get("values")
    gap = (pt - vl).dropna() if vl is not None else pd.Series(dtype=float)
    print(f"\n{tag}: {len(w)} panels, {int(d.n_cols_missing.iloc[0])} columns with missing, "
          f"{int(d.n_panels.iloc[0])} panels defined")
    print(f"  median panels_tree {pt.median():.3f}   median values {vl.median():.3f}   "
          f"median per-panel gap {gap.median():.3f}")
    ok[f"{tag} N1 panels_tree >= 0.80"] = bool(pt.median() >= 0.80)
    ok[f"{tag} N2 values < 0.60"] = bool(vl.median() < 0.60)
    ok[f"{tag} N3 gap >= 0.20"] = bool(gap.median() >= 0.20)
print()
for k, v in ok.items(): print(("  HELD    " if v else "  FAILED  ") + k)
print("\n" + ("ALL HELD" if all(ok.values()) else "NOT ALL HELD"))
sys.exit(0 if all(ok.values()) else 1)
