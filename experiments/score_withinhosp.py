"""Score WITHINHOSP against prereg/WITHINHOSP.md."""
import pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1] / "results/withinhosp"
ok = {}
for tag, w1 in (("eicu", 0.85), ("mimic4", 0.75)):
    f = R / f"{tag}.csv"
    if not f.exists(): print(f"{tag}: NOT YET RUN"); continue
    d = pd.read_csv(f)
    pool, site = d[d.site == "POOLED"], d[d.site != "POOLED"]
    nsites = site.site.nunique()
    mp, mw = site.pooled_on_site.median(), site.within_site.median()
    mv = site.values_on_site.median()
    print(f"\n{tag}: {pool.panel.nunique()} panels, {nsites} sites contributing")
    print(f"  pooled fit on all holdout rows : median {pool.pooled_on_site.median():.3f}")
    print(f"  values, all holdout rows       : median {pool.values_on_site.median():.3f}")
    print(f"  pooled fit, evaluated per site : median {mp:.3f}")
    print(f"  WITHIN-site fit                : median {mw:.3f}")
    print(f"  values, evaluated per site     : median {mv:.3f}")
    ok[f"{tag} W1 within-site >= {w1}"] = bool(mw >= w1)
    ok[f"{tag} W2 within 0.10 of pooled-everywhere"] = bool(abs(mw - pool.pooled_on_site.median()) <= 0.10)
    ok[f"{tag} W3 within-site values < 0.60"] = bool(mv < 0.60)
    if tag == "eicu": ok["eicu W5 >= 20 sites contribute"] = bool(nsites >= 20)
print()
for k, v in ok.items(): print(("  HELD    " if v else "  FAILED  ") + k)
print("\n" + ("ALL HELD" if all(ok.values()) else "NOT ALL HELD"))
sys.exit(0 if all(ok.values()) else 1)
