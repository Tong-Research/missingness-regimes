"""Score prereg/SMDI.md: single-model block importance versus separate-block AUROC."""
import glob, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; V = lambda ok: "HOLDS" if ok else "FAILS"
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "results/smdi/*.csv")))], ignore_index=True)
d = d[d.panel >= 0].copy()
g = d.groupby(["dataset", "panel"]).agg(av=("auroc_values", "first"), ap=("auroc_panels", "first"), ab=("auroc_both", "first"),
                                        share=("imp_share_values", "mean"), share_sd=("imp_share_values", "std"),
                                        flips=("imp_share_values", lambda s: int(len(set(np.sign(s.dropna() - 0.5))) > 1))).dropna(subset=["av", "ap"])
print(f"=== SMDI ({d.dataset.nunique()}/17 datasets, {len(g)} panels with both blocks) ===")
p1 = int(((g.share <= 0.10) & (g.av >= 0.80)).sum() + ((g.share >= 0.90) & (g.ap >= 0.80)).sum())
print(f"P1 panels where a block with standalone AUROC >= 0.80 gets <= 10% of the importance: {p1} (needs >= 10): {V(p1 >= 10)}")
ds = g.groupby("dataset").agg(share=("share", "mean"), av=("av", "mean"), ap=("ap", "mean"))
dis = int((np.sign(ds.share - 0.5) != np.sign(ds.av - ds.ap)).sum())
print(f"P2 dataset-level regime call disagrees on {dis}/{len(ds)} (needs >= 2): {V(dis >= 2)}")
close = g[(g.av - g.ap).abs() <= 0.05]; fr = close.flips.mean() if len(close) else np.nan
print(f"P3 sign flips across seeds on {fr:.0%} of the {len(close)} near-tied panels (needs >= 20%): {V(len(close) > 0 and fr >= 0.20)}")
comp = (g.ab >= g[["av", "ap"]].max(axis=1) + 0.01).mean()
print(f"P4 single model beats the better block by >= 0.01 on {comp:.0%} of panels (needs >= 30%): {V(comp >= 0.30)}")
print("\nper dataset (mean over panels):")
print(pd.concat([ds.round(3), g.groupby("dataset").size().rename("panels"),
                 g.groupby("dataset").flips.mean().round(2).rename("flip_rate")], axis=1).to_string())
