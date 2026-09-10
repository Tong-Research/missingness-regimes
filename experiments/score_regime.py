"""Score prereg/REGIME.md from results/regime/*.csv."""
import glob, pathlib, numpy as np, pandas as pd
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parents[1]; V = lambda ok: "HOLDS" if ok else "FAILS"
s = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results/regime/site_*.csv"))]); print("=== REGIME site fingerprint ==="); print(s.round(3).to_string(index=False))
e = s[(s.db == "eicu") & (s.subset == "top20")].set_index("features")
if len(e): print(f"P1 eICU top-20: mask/majority = {e.loc['mask','ratio']:.2f} (>= 5), mask/values = {e.loc['mask','accuracy']/e.loc['values','accuracy']:.2f} (>= 0.8): {V(e.loc['mask','ratio'] >= 5 and e.loc['mask','accuracy'] >= 0.8 * e.loc['values','accuracy'])}")
t = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results/regime/transfer_*.csv"))]); t = t[t.get("note", pd.Series(index=t.index, dtype=object)).isna()] if "note" in t else t
print(f"\n=== REGIME transfer ({len(t)} hospitals) ===")
p2 = int((t.gain_adapted >= 0).sum()); print(f"P2 adapted >= transfer on {p2}/{len(t)} (>= 14), median gain {t.gain_adapted.median():+.4f} (>= 0.005): {V(p2 >= 14 and t.gain_adapted.median() >= 0.005)}")
rho = stats.spearmanr(t.gain_adapted, t.regime_distance)[0]; print(f"P3 Spearman(gain, regime_distance) = {rho:+.3f} (>= 0.4): {V(rho >= 0.4)}")
p4 = int((t.mask_only_transfer < t.mask_only_in).sum()); print(f"P4 mask-only transfer < in-site on {p4}/{len(t)} (>= 14): {V(p4 >= 14)}")
print(f"reported: delta_transfer median {t.delta_transfer.median():+.4f}, transfer > in_site on {int((t.delta_transfer > 0).sum())}/{len(t)}")
print(t.sort_values("regime_distance")[["site", "n_T", "pos_T", "regime_distance", "in_site", "transfer", "adapted", "gain_adapted", "mask_only_in", "mask_only_transfer"]].round(4).to_string(index=False))
