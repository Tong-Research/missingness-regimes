"""WITHINHOSP (prereg/WITHINHOSP.md): is the latent class that generates the indicator dependence
the hospital, or only correlated with it?

Paper A reports that a Bernoulli mixture over the mask reproduces an unrestricted learner's
indicator-indicator dependence, and that the recovered class matches the hospital at 5.82x the
majority baseline with adjusted mutual information 0.382. An AMI of 0.382 is well short of 1, so the
class is not simply the hospital, and the paper should say how far short in a way a reader can check.

The direct test is to condition on the hospital and look again. If the whole dependence is the
between-hospital regime, then within one hospital there is nothing left for a panel's presence to
say about another panel's presence, and the within-site AUROC collapses toward 0.5.

Three quantities per site, on the SAME panels, which are defined once on the pooled dev half so that
the comparison is not confounded by different panel definitions:

  pooled_on_site   the pooled-fit tree, evaluated on this site's holdout rows
  within_site      a tree fit on this site's dev rows only, evaluated on this site's holdout rows
  values_on_site   the always-observed values, fit pooled, evaluated on this site's holdout rows

`within_site` is the headline. Sites are used only to partition rows; no model is ever given the
site id.
"""
import argparse, csv, pathlib, sys, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from probe_unseen import load_full, dev_mask
from probe_common import tree
from probe_explore import keepcols
from sklearn.metrics import roc_auc_score
C = pathlib.Path.home() / ".cache/phd-matrices"

def auc(y, s): return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", required=True, help="e.g. cand_eicu_regime")
    a.add_argument("--sites", required=True, help="e.g. eicu -> eicu_sites.npz")
    a.add_argument("--min-rows", type=int, default=1000, help="smallest site kept, counted on the full matrix")
    a.add_argument("--out", required=True)
    a = a.parse_args()

    Xd, Xh = load_full(a.dataset)
    g = np.asarray(np.load(C / f"{a.sites}_sites.npz", allow_pickle=True)["g"]).ravel()
    dm = dev_mask(a.dataset, len(g))
    assert len(g) == len(Xd) + len(Xh), f"site vector {len(g)} != {len(Xd)}+{len(Xh)}"
    gd, gh = g[dm], g[~dm]

    keep = ~(np.isnan(Xd).all(0) & np.isnan(Xh).all(0)); Xd, Xh = Xd[:, keep], Xh[:, keep]
    Md, Mh = np.isnan(Xd), np.isnan(Xh); rate = Md.mean(0); p = Xd.shape[1]
    mc = [j for j in range(p) if 0.01 <= rate[j] <= 0.99]; full = [j for j in range(p) if rate[j] < 0.01]

    # Panels exactly as in probe_rrstruct: Jaccard >= 0.9 clusters on the pooled dev half.
    clus = {}
    for j in mc:
        for k in clus:
            inter = (Md[:, j] & Md[:, k]).sum(); uni = (Md[:, j] | Md[:, k]).sum()
            if uni and inter / uni >= 0.9: clus[j] = clus[k]; break
        else: clus[j] = j
    panels = {}
    for j in mc: panels.setdefault(clus[j], []).append(j)
    reps = [sorted(v, key=lambda j: abs(rate[j] - np.median(rate[v])))[0] for v in panels.values()]
    Pd, Ph = (~Md[:, reps]).astype(float), (~Mh[:, reps]).astype(float); B = len(reps)

    counts = {s: int((g == s).sum()) for s in np.unique(g)}
    sites = sorted([s for s, n in counts.items() if n >= a.min_rows])
    print(f"{a.dataset}: {p} cols, {B} panels, {len(np.unique(g))} sites, "
          f"{len(sites)} with >= {a.min_rows} rows", flush=True)

    rows = []
    for b in range(B):
        yd_all, yh_all = 1 - Pd[:, b], 1 - Ph[:, b]
        if not (0.01 <= yd_all.mean() <= 0.99 and 0.01 <= yh_all.mean() <= 0.99): continue
        oth = [c for c in range(B) if c != b]
        if not oth: continue
        pooled = tree().fit(Pd[:, oth], yd_all)
        vmod = None
        if full:
            kc = keepcols(Xd[:, full]); vmod = (tree().fit(Xd[:, full][:, kc], yd_all), kc)
        # the pooled reference, on all holdout rows, so the site numbers have something to sit against
        rows.append(dict(dataset=a.dataset, site="POOLED", n_site=len(Ph), panel=b,
                         base_rate=float(yh_all.mean()), n_panels=B,
                         pooled_on_site=auc(yh_all, pooled.predict_proba(Ph[:, oth])[:, 1]),
                         within_site=np.nan,
                         values_on_site=(auc(yh_all, vmod[0].predict_proba(Xh[:, full][:, vmod[1]])[:, 1]) if vmod else np.nan)))
        for s in sites:
            sd, sh = gd == s, gh == s
            if sh.sum() < 100 or sd.sum() < 100: continue
            yh = yh_all[sh]; yd = yd_all[sd]
            if len(np.unique(yh)) < 2: continue
            r = dict(dataset=a.dataset, site=str(s), n_site=int(sh.sum()), panel=b,
                     base_rate=float(yh.mean()), n_panels=B,
                     pooled_on_site=auc(yh, pooled.predict_proba(Ph[sh][:, oth])[:, 1]),
                     within_site=np.nan, values_on_site=np.nan)
            if len(np.unique(yd)) == 2:
                r["within_site"] = auc(yh, tree().fit(Pd[sd][:, oth], yd).predict_proba(Ph[sh][:, oth])[:, 1])
            if vmod is not None:
                r["values_on_site"] = auc(yh, vmod[0].predict_proba(Xh[sh][:, full][:, vmod[1]])[:, 1])
            rows.append(r)
        done = [r for r in rows if r["panel"] == b and r["site"] != "POOLED"]
        pool = [r for r in rows if r["panel"] == b and r["site"] == "POOLED"][0]
        med = float(np.nanmedian([r["within_site"] for r in done])) if done else float("nan")
        medp = float(np.nanmedian([r["pooled_on_site"] for r in done])) if done else float("nan")
        print(f"  panel {b:3d} rate {yh_all.mean():.3f} | pooled-everywhere {pool['pooled_on_site']:.3f} "
              f"values {pool['values_on_site']:.3f} | {len(done)} sites: median pooled-on-site {medp:.3f}, "
              f"median within-site {med:.3f}", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")

if __name__ == "__main__": main()
