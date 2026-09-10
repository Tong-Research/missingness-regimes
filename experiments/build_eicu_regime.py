"""SHOWCASE.md item G: eICU first-unit-stay TREATMENT-REGIME matrix -> ~/.cache/phd-matrices/cand_eicu_regime.npz
Rule as declared in the 16:05 amendment. Row order = eicu_sites._cohort(), asserted against the cached eicu_sites.npz."""
import pathlib, re, sys, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import eicu_sites as es
CACHE = pathlib.Path.home() / ".cache/phd-matrices"; OUT = CACHE / "cand_eicu_regime.npz"; W = es.WINDOW_MIN
DRUGS = ["norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine", "dobutamine", "insulin"]

def main():
    t0 = time.time(); z = np.load(CACHE / "eicu_sites.npz", allow_pickle=True); X0, y0 = np.asarray(z["X"], float), np.asarray(z["y"]).ravel().astype(int)
    p = pd.read_csv(es.ROOT / "patient.csv.gz", usecols=["patientunitstayid", "hospitalid", "unittype", "age", "gender", "unitvisitnumber", "unitdischargeoffset", "hospitaldischargestatus"],
                    dtype={"age": str, "gender": str, "unittype": str, "hospitaldischargestatus": str}, low_memory=False)
    p = p[(p["unitvisitnumber"] == 1) & (p["unitdischargeoffset"] >= es.WINDOW_MIN)]
    p["age_n"] = pd.to_numeric(p["age"].replace("> 89", "90"), errors="coerce"); p = p[p["age_n"] >= 18].dropna(subset=["hospitaldischargestatus"])
    p["died"] = (p["hospitaldischargestatus"] == "Expired").astype(int)
    keep = p["hospitalid"].value_counts().loc[lambda s_: s_ >= es.MIN_SITE_N].index; coh = p[p["hospitalid"].isin(keep)].reset_index(drop=True)
    assert len(coh) == len(X0), (len(coh), len(X0))
    assert np.array_equal(coh["died"].to_numpy().astype(int), y0) and np.allclose(coh["age_n"].to_numpy(), X0[:, -2]), "cohort order differs from the cached matrix"
    n = len(coh); sids = set(coh["patientunitstayid"]); six = {s: i for i, s in enumerate(coh["patientunitstayid"])}
    print(f"  cohort aligned: {n:,} stays [{time.time()-t0:.0f}s]", flush=True)
    # ventilator settings: seven most frequent numeric labels in respFlowSettings within the window
    parts = []
    for ch in pd.read_csv(es.ROOT / "respiratoryCharting.csv.gz", usecols=["patientunitstayid", "respchartoffset", "respcharttypecat", "respchartvaluelabel", "respchartvalue"], dtype={"patientunitstayid": "int64", "respchartoffset": "float64", "respcharttypecat": str, "respchartvaluelabel": str, "respchartvalue": str}, chunksize=3_000_000):
        ch = ch[(ch.respcharttypecat == "respFlowSettings") & ch.patientunitstayid.isin(sids) & (ch.respchartoffset >= 0) & (ch.respchartoffset <= W)]
        ch = ch.assign(v=pd.to_numeric(ch.respchartvalue.astype(str).str.replace("%", "", regex=False), errors="coerce")).dropna(subset=["v"])
        parts.append(ch[["patientunitstayid", "respchartvaluelabel", "v"]])
    rc = pd.concat(parts, ignore_index=True); labels = rc.respchartvaluelabel.value_counts().index[:7].tolist(); print(f"  ventilator labels: {labels}", flush=True)
    vent = np.full((n, len(labels)), np.nan); m = rc[rc.respchartvaluelabel.isin(labels)].groupby(["patientunitstayid", "respchartvaluelabel"])["v"].mean()
    for (s, lab), v in m.items(): vent[six[s], labels.index(lab)] = v
    # infusions
    inf = pd.read_csv(es.ROOT / "infusionDrug.csv.gz", usecols=["patientunitstayid", "infusionoffset", "drugname", "drugrate"], dtype={"drugname": str, "drugrate": str}, low_memory=False)
    inf = inf[inf.patientunitstayid.isin(sids) & (inf.infusionoffset >= 0) & (inf.infusionoffset <= W)]; inf["rate"] = pd.to_numeric(inf.drugrate, errors="coerce"); inf = inf.dropna(subset=["rate"])
    name = inf.drugname.astype(str).str.lower(); vaso = np.full((n, len(DRUGS)), np.nan)
    for j, d in enumerate(DRUGS):
        sub = inf[name.str.contains(d, regex=False)]
        for s, r in sub.groupby("patientunitstayid")["rate"].max().items(): vaso[six[s], j] = r
    # renal replacement
    tr = pd.read_csv(es.ROOT / "treatment.csv.gz", usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"], dtype={"treatmentstring": str}, low_memory=False)
    tr = tr[tr.patientunitstayid.isin(sids) & (tr.treatmentoffset >= 0) & (tr.treatmentoffset <= W) & tr.treatmentstring.astype(str).str.contains("dialysis|CRRT", case=False, regex=True)]
    rrt = np.full((n, 1), np.nan)
    for s, off in tr.groupby("patientunitstayid")["treatmentoffset"].min().items(): rrt[six[s], 0] = off / 60.0
    X = np.hstack([X0, vent, vaso, rrt]); cols = [f"lab_{i}" for i in range(X0.shape[1] - 2)] + ["age", "male"] + [f"vent_{re.sub(r'[^A-Za-z0-9]+', '_', l)}" for l in labels] + [f"inf_{d}" for d in DRUGS] + ["rrt_first_hour"]
    np.savez_compressed(OUT, X=X, y=y0, cols=np.array(cols)); M = np.isnan(X)
    print(f"  eicu_regime: n {n:,} d {X.shape[1]} prev {y0.mean():.3f} missing {M.mean():.1%} patterns {len({r.tobytes() for r in M}):,}; ventilated {int((~np.isnan(vent)).any(1).sum()):,}, any infusion {int((~np.isnan(vaso)).any(1).sum()):,}, RRT {int((~np.isnan(rrt)).any(1).sum()):,} -> {OUT} [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
