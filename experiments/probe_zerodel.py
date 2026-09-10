"""ZERODEL (prereg/ZERODEL.md): does the ICLR 2024 dropout correction survive a latent regime?

Dai et al. (ICLR 2024 oral, arXiv:2403.15500) state their contribution as a theorem, in one sentence
on their own repository front page:

    "conditional independence (CI) relations in the data with dropouts, after deleting the samples
     with zero values for conditioned variables, are identical to the CI relations in the original
     data."

That holds under their assumptions, of which (A3) is that a variable's dropout is directly affected
only by that variable's own value -- not by other variables' values or dropouts. Our regime violates
(A3): the mask is driven by a latent institution, so it is affected by nothing the model sees.

This tests the theorem's own conclusion, with their code, unmodified. For many random triples
(X, Y, Z) we ask whether the CI verdict on the complete data agrees with the verdict their rule
returns on the masked data. Their theorem says the two should agree.

Two deletion rules are compared, both theirs:
  cond  delete rows where any CONDITIONING variable is zero -- their default, the theorem's rule
  all   delete rows where any INVOLVED variable is zero -- the alternative their own docstring
        offers, which is ordinary test-wise deletion and which they note is "still correct ...
        though it may be less powerful"

Missing values are encoded as 0, which is their dropout convention. Our variables are Gaussian, so
an exact zero has measure zero and "is zero" coincides with "is missing".
"""
import argparse, csv, json, pathlib, sys, warnings, numpy as np
warnings.simplefilter("ignore")

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True, help="npz from export_plasmode.py")
    a.add_argument("--out", required=True)
    a.add_argument("--alpha", type=float, default=0.01)
    a.add_argument("--triples", type=int, default=300)
    a.add_argument("--maxcond", type=int, default=3)
    a = a.parse_args()

    from csl.utils.cit import CIT

    z = np.load(a.data, allow_pickle=True); meta = json.loads(str(z["meta"])); P = int(meta["p"])
    rng = np.random.default_rng(20260907)
    triples = []
    while len(triples) < a.triples:
        x, y = rng.choice(P, 2, replace=False)
        k = int(rng.integers(0, a.maxcond + 1))
        pool = [q for q in range(P) if q not in (x, y)]
        cond = sorted(rng.choice(pool, k, replace=False).tolist()) if k else []
        triples.append((int(x), int(y), cond))

    rows = []
    for cell in meta["cells"]:
        key = cell["key"]
        X = np.asarray(z[f"{key}_X"], dtype=np.float64)
        Xm = np.asarray(z[f"{key}_Xm"], dtype=np.float64)
        Xz = np.nan_to_num(Xm, nan=0.0)              # their dropout convention

        truth = CIT(X, method="fisherz")
        theirs = CIT(Xz, method="zerodel_fisherz")   # their rule: delete on the conditioning set
        # their documented alternative, one line as their docstring describes
        allvar = CIT(Xz, method="zerodel_fisherz")
        allvar._get_index_no_mv_rows_orig = allvar._get_index_no_mv_rows

        agree_c = agree_a = 0
        for (x, y, cond) in triples:
            t = truth(x, y, cond) > a.alpha                       # independent on complete data?
            c = theirs(x, y, cond) > a.alpha
            # delete on ALL involved variables by asking their own helper for the full set
            import types
            keep = np.where(np.all(allvar.nonzero_boolean_mat[:, [x, y] + cond], axis=1))[0]
            if len(keep) > len(cond) + 3:
                sub = CIT(Xz[keep], method="fisherz")
                aa = sub(x, y, cond) > a.alpha
            else:
                aa = True
            agree_c += int(t == c); agree_a += int(t == aa)
        rows.append(dict(rep=meta["rep"], gamma=cell["gamma"], alignment=cell["alignment"],
                         n_triples=len(triples), alpha=a.alpha,
                         agree_cond=agree_c / len(triples), agree_all=agree_a / len(triples)))
        print(f"  gamma {cell['gamma']} {cell['alignment']:7s}: "
              f"their rule agrees with the complete-data verdict on {agree_c/len(triples):.3f}, "
              f"delete-on-all {agree_a/len(triples):.3f}", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")

if __name__ == "__main__": main()
