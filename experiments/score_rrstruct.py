"""Score prereg/RRSTRUCT.md and prereg/MASKINC.md from results/{rrstruct,maskinc}/*.csv."""
import glob, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; V = lambda ok: "HOLDS" if ok else "FAILS"
REG = ["cand_eicu_regime", "cand_mimic4_regime", "cand_nhanes", "sweep_42739"]
def rd(n):
    fs = sorted(glob.glob(str(ROOT / f"results/{n}/*.csv"))); return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

def rrstruct():
    d = rd("rrstruct"); au = d[d.kind == "auroc"]
    if not len(au): print("no rrstruct rows"); return
    m = au.groupby(["dataset", "model"]).value.mean().unstack("model"); npan = d.groupby("dataset").n_panels.first()
    lcaK = {int(c.split("_")[1]): c for c in m.columns if c.startswith("lca_")}
    low = [lcaK[k] for k in sorted(lcaK) if k <= 20 and lcaK[k] in m]; best20 = m[low].max(axis=1); allk = [lcaK[k] for k in sorted(lcaK)]
    print(f"\n=== RRSTRUCT ({len(m)}/17 datasets) ===")
    r1 = best20 / m.panels_tree; p1 = int((r1 >= 0.95).sum()); print(f"P1 best mixture K<=20 >= 0.95 x tree on {p1}/{len(m)} (needs >= 12): {V(p1 >= 12)}")
    s = d[d.kind == "sites_purity"]; a_ = d[d.kind == "sites"]
    if len(s):
        pe = s[s.dataset == "cand_eicu_regime"].value.max(); ae = a_[a_.dataset == "cand_eicu_regime"].value.max(); pm = s[s.dataset == "cand_mimic4_regime"].value.max()
        print(f"P2 eICU purity/majority {pe:.2f} (>= 3) AMI {ae:.3f} (>= 0.15); MIMIC purity {pm:.2f} (>= 1.5): {V(pe >= 3 and ae >= 0.15 and pm >= 1.5)}")
    p3a = int((m.panels_l1 < 0.98 * m.panels_tree).sum()); p3b = int((best20 > m.panels_l1).sum()); print(f"P3 L1 < 0.98 x tree on {p3a}/{len(m)} (needs >= 8) and mixture > L1 on {p3b}/{len(m)} (needs >= 12): {V(p3a >= 8 and p3b >= 12)}")
    g = m.reindex(REG); p4 = bool((g["values"] <= 0.70).all() and (g.panels_tree >= 0.85).all() and ((g["both"] - g["values"]) >= 0.15).all())
    print(f"P4 on the four regime sets: values <= 0.70 ({int((g['values'] <= 0.70).sum())}/4), panels_tree >= 0.85 ({int((g.panels_tree >= 0.85).sum())}/4), both - values >= 0.15 ({int(((g['both'] - g['values']) >= 0.15).sum())}/4): {V(p4)}")
    need = {}
    for ds in m.index:
        t = m.loc[ds, "panels_tree"]; ks = [k for k in sorted(lcaK) if m.loc[ds, lcaK[k]] >= 0.95 * t]; need[ds] = min(ks) if ks else np.inf
    ns = pd.Series(need); p5 = int((ns <= 20).sum()); print(f"P5 smallest K reaching 95% of tree is <= 20 on {p5}/{len(m)} (needs >= 12); of those, {int(((ns <= 20) & (npan >= 15)).sum())} have >= 15 panels: {V(p5 >= 12)}")
    ka = int((m.panels_tree < 0.70).sum()); print(f"kill (a): panels_tree < 0.70 on {ka}/{len(m)} (kills at >= 6): {V(ka < 6)}")
    out = pd.concat([m[["values", "panels_tree", "panels_l1", "both"]] if "values" in m else m[["panels_tree", "panels_l1"]], best20.rename("lca_best20"), r1.rename("ratio"), ns.rename("K_needed"), npan.rename("panels")], axis=1)
    print(out.round(3).to_string())

def maskinc():
    d = rd("maskinc")
    if not len(d): print("\nno maskinc rows"); return
    w = d[d.arm != "mask_full"].groupby(["dataset", "q", "arm"]).auprc.median().unstack("arm")
    gf = (w["values+mask_full"] - w["values"]).unstack("q"); gk = (w["values+mask_kept"] - w["values"]).unstack("q")
    print(f"\n=== MASKINC ({len(gf)}/17 datasets) ===")
    p1 = int((gf[1.0] <= 0.005).sum()); print(f"P1 gain_full at q=1.0 <= 0.005 on {p1}/{len(gf)} (needs all 17): {V(p1 == len(gf) == 17)}")
    p2 = int((gf[0.1] >= 0.01).sum()); print(f"P2 gain_full at q=0.1 >= 0.01 on {p2}/{len(gf)} (needs >= 10): {V(p2 >= 10)}")
    med = gf.median(); mono = bool(np.all(np.diff(med.reindex([0.1, 0.25, 0.5, 0.75, 1.0]).values) <= 1e-9)); print(f"P3 median gain_full by q: " + " ".join(f"{q}:{med[q]:+.4f}" for q in [0.1, 0.25, 0.5, 0.75, 1.0]) + f" non-increasing: {V(mono)}")
    p4 = int((gf.reindex(REG)[0.1] >= 0.02).sum()); print(f"P4 gain_full at q=0.1 >= 0.02 on {p4}/4 regime sets: {V(p4 == 4)}")
    p5 = int((gk[0.1] < gf[0.1]).sum()); print(f"P5 gain_kept < gain_full at q=0.1 on {p5}/{len(gf)} (needs >= 12): {V(p5 >= 12)}")
    print(pd.concat([gf.add_prefix("full_q"), gk[[0.1, 0.5]].add_prefix("kept_q")], axis=1).round(4).to_string())

if __name__ == "__main__":
    for fn in (rrstruct, maskinc):
        try: fn()
        except Exception as e: print(f"[{fn.__name__}] not scorable yet: {type(e).__name__}: {e}")
