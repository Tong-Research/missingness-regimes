"""SHOWCASE.md item F: MIMIC-IV first-ICU-stay TREATMENT-REGIME matrix -> ~/.cache/phd-matrices/cand_mimic4_regime.npz

The published matrix (mimic4_sites.npz: 40 most-frequent first-24h labs + age + male, in-hospital mortality) is extended
with fields that exist only when a treatment occurred in the first 24 h, left NaN otherwise:
  ventilator settings from chartevents: PEEP 220339, FiO2 223835, tidal volume observed 224685, set respiratory rate
    224688, plateau pressure 224696, mean airway pressure 224697, minute volume 224687 (mean over the window);
  vasopressor / inotrope rates from inputevents (max rate in window): norepinephrine 221906, epinephrine 221289,
    vasopressin 222315, phenylephrine 221749, dopamine 221662, dobutamine 221653; insulin infusion 223258;
  renal replacement from procedureevents: hours of CRRT/dialysis (225802, 225803, 225441, 225955, 225809) in the window.
Cohort and row order are those of mimic4_sites._build (asserted against the cached npz), so the labels are the published
ones; the hash split key "cand_mimic4_regime" makes its halves independent of every earlier MIMIC probe.
"""
import pathlib, sys, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mimic4_sites as ms
CACHE = pathlib.Path.home() / ".cache/phd-matrices"; OUT = CACHE / "cand_mimic4_regime.npz"
VENT = {220339: "peep", 223835: "fio2", 224685: "tidal_volume", 224688: "set_resp_rate", 224696: "plateau_pressure", 224697: "mean_airway_pressure", 224687: "minute_volume"}
VASO = {221906: "norepinephrine", 221289: "epinephrine", 222315: "vasopressin", 221749: "phenylephrine", 221662: "dopamine", 221653: "dobutamine", 223258: "insulin_infusion"}
RRT = {225802, 225803, 225441, 225955, 225809}

def cohort():
    icu = pd.read_csv(ms.ROOT / "icu" / "icustays.csv.gz", usecols=["hadm_id", "stay_id", "first_careunit", "intime", "los"], parse_dates=["intime"])
    icu = icu.sort_values("intime").drop_duplicates("hadm_id", keep="first"); icu = icu[icu["los"] >= 1.0]
    adm = pd.read_csv(ms.ROOT / "hosp" / "admissions.csv.gz", usecols=["hadm_id", "subject_id", "admittime", "hospital_expire_flag"], parse_dates=["admittime"])
    pat = pd.read_csv(ms.ROOT / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_age", "anchor_year", "gender"])
    coh = icu.merge(adm, on="hadm_id").merge(pat, on="subject_id")
    coh["AGE"] = (coh["anchor_age"] + (coh["admittime"].dt.year - coh["anchor_year"])).astype(float)
    coh = coh[(coh["AGE"] >= 18) & (coh["AGE"] <= 100)]
    keep = coh["first_careunit"].value_counts().loc[lambda s: s >= ms.MIN_UNIT_N].index.tolist()
    return coh[coh["first_careunit"].isin(keep)].reset_index(drop=True)

def main():
    t0 = time.time(); z = np.load(CACHE / "mimic4_sites.npz", allow_pickle=True); X0, y0, cols0 = np.asarray(z["X"], float), np.asarray(z["y"]).ravel().astype(int), [str(c) for c in z["cols"]]
    coh = cohort(); assert len(coh) == len(X0), (len(coh), len(X0))
    assert np.array_equal(coh["hospital_expire_flag"].to_numpy().astype(int), y0) and np.allclose(coh["AGE"].to_numpy(), X0[:, cols0.index("age")]), "cohort order differs from the cached matrix"
    print(f"  cohort aligned: {len(coh):,} stays [{time.time()-t0:.0f}s]", flush=True)
    stays = set(coh["stay_id"]); six = {s: i for i, s in enumerate(coh["stay_id"])}; intime = dict(zip(coh["stay_id"], coh["intime"]))
    n = len(coh); vent_sum = np.zeros((n, len(VENT))); vent_n = np.zeros((n, len(VENT))); vidx = {k: j for j, k in enumerate(VENT)}
    reader = pd.read_csv(ms.ROOT / "icu" / "chartevents.csv.gz", usecols=["stay_id", "itemid", "charttime", "valuenum"], parse_dates=["charttime"], chunksize=5_000_000)
    for ci, ch in enumerate(reader):
        ch = ch[ch["itemid"].isin(VENT.keys()) & ch["stay_id"].isin(stays)].dropna(subset=["valuenum"])
        if len(ch):
            hrs = (ch["charttime"] - ch["stay_id"].map(intime)).dt.total_seconds() / 3600.0; ch = ch[(hrs >= 0) & (hrs <= ms.WINDOW_H)]
            for s, it, v in zip(ch["stay_id"].to_numpy(), ch["itemid"].to_numpy(), ch["valuenum"].to_numpy()):
                vent_sum[six[s], vidx[it]] += v; vent_n[six[s], vidx[it]] += 1
        if ci % 10 == 0: print(f"    chartevents chunk {ci}: {int((vent_n.sum(1) > 0).sum()):,} stays with any ventilator setting [{time.time()-t0:.0f}s]", flush=True)
    vent = np.where(vent_n > 0, vent_sum / np.maximum(vent_n, 1), np.nan)
    inp = pd.read_csv(ms.ROOT / "icu" / "inputevents.csv.gz", usecols=["stay_id", "itemid", "starttime", "rate"], parse_dates=["starttime"])
    inp = inp[inp["itemid"].isin(VASO.keys()) & inp["stay_id"].isin(stays)].dropna(subset=["rate"])
    hrs = (inp["starttime"] - inp["stay_id"].map(intime)).dt.total_seconds() / 3600.0; inp = inp[(hrs >= 0) & (hrs <= ms.WINDOW_H)]
    vaso = np.full((n, len(VASO)), np.nan); aidx = {k: j for j, k in enumerate(VASO)}
    for (s, it), r in inp.groupby(["stay_id", "itemid"])["rate"].max().items(): vaso[six[s], aidx[it]] = r
    pe = pd.read_csv(ms.ROOT / "icu" / "procedureevents.csv.gz", usecols=["stay_id", "itemid", "starttime", "endtime"], parse_dates=["starttime", "endtime"])
    pe = pe[pe["itemid"].isin(RRT) & pe["stay_id"].isin(stays)]
    st = (pe["starttime"] - pe["stay_id"].map(intime)).dt.total_seconds() / 3600.0; en = (pe["endtime"] - pe["stay_id"].map(intime)).dt.total_seconds() / 3600.0
    dur = (np.minimum(en, ms.WINDOW_H) - np.maximum(st, 0)).clip(lower=0); pe = pe.assign(h=dur)[dur > 0]
    rrt = np.full((n, 1), np.nan)
    for s, h in pe.groupby("stay_id")["h"].sum().items(): rrt[six[s], 0] = h
    X = np.hstack([X0, vent, vaso, rrt]); cols = cols0 + list(VENT.values()) + list(VASO.values()) + ["rrt_hours"]
    np.savez_compressed(OUT, X=X, y=y0, cols=np.array(cols)); M = np.isnan(X)
    print(f"  mimic4_regime: n {len(X):,} d {X.shape[1]} prev {y0.mean():.3f} missing {M.mean():.1%} patterns {len({r.tobytes() for r in M}):,}; ventilated {int((vent_n.sum(1)>0).sum()):,}, any vasopressor {int((~np.isnan(vaso)).any(1).sum()):,}, RRT {int((~np.isnan(rrt)).any(1).sum()):,} -> {OUT} [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
