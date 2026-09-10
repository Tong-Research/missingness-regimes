"""Figures for the merged regimes paper, generated from the committed results.

Emits pgfplots source rather than PDFs so the figures typeset in the document's
own fonts and share the house style in ../../common/pgfstyle.tex.

Run: ../paper-plco-hypergraph/.venv/bin/python figures/make_figures.py
"""
import glob
import pathlib
import sys

import pandas as pd

SRC = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).resolve().parent
# papers/common/, found by searching upward: a fixed hop count is right in the
# monorepo and wrong in an extracted release
for _d in [pathlib.Path(__file__).resolve(), *pathlib.Path(__file__).resolve().parents]:
    if (_d / "common" / "pgfemit.py").exists():
        sys.path.insert(0, str(_d / "common")); break
else:
    raise SystemExit("could not find common/pgfemit.py above " + __file__)

from pgfemit import Fig, declutter, fmt   # noqa: E402

NAME = {"cand_eicu_regime": "eICU", "cand_mimic4_regime": "MIMIC-IV",
        "cand_nhanes": "NHANES", "sweep_42739": "road safety",
        "cand_acs_income": "ACS income", "cand_airbnb": "Airbnb",
        "cand_higgs": "Higgs", "cand_porto": "Porto"}
REG = ["cand_eicu_regime", "cand_mimic4_regime", "cand_nhanes", "sweep_42739"]
STR = ["cand_acs_income", "cand_airbnb", "cand_higgs",
       "sweep_42737", "sweep_46654", "sweep_46703"]

# group -> (label, pgfplots series style)
GROUP = {
    "protocol-driven": ("protocol-driven", "hgred, mark=*, mark size=2.1pt, "
                        "mark options={fill=hgred, draw=white, line width=0.3pt}"),
    "structural":      ("structural", "hgblue, mark=square*, mark size=2.0pt, "
                        "mark options={fill=hgblue, draw=white, line width=0.3pt}"),
    "unclassified":    ("unclassified", "hggrey, mark=triangle*, mark size=2.4pt, "
                        "mark options={fill=hggrey, draw=white, line width=0.3pt}"),
}
LO, HI = 0.45, 1.06


def nm(d):
    return NAME.get(d, d.replace("sweep_", ""))


def grp(ds):
    return "protocol-driven" if ds in REG else "structural" if ds in STR else "unclassified"


def panels(sub):
    fs = sorted(glob.glob(str(SRC / f"results/{sub}/*.csv")))
    if not fs:
        raise SystemExit(f"no results under results/{sub}/")
    x = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return x[x.kind == "auroc"].groupby(["dataset", "model"]).value.mean().unstack("model")


def _scatter_panel(ax, m, first, offsets):
    """One dissociation panel: mask-from-mask against mask-from-values.

    ``offsets`` maps dataset to a label offset.  Both panels are given the same
    map: the mirror is meant to be read against panel A, and a label that jumps
    from one side of its marker to the other between the two panels makes that
    comparison harder than it needs to be.
    """
    ax.raw(f"\\addplot[hg rule, forget plot] coordinates "
           f"{{({LO},{LO}) ({HI},{HI})}};")
    for key, (label, style) in GROUP.items():
        rows = [ds for ds in m.index if grp(ds) == key]
        if not rows:
            continue
        ax.plot(f"{style}, only marks",
                [m.loc[ds, "values"] for ds in rows],
                [m.loc[ds, "panels_tree"] for ds in rows],
                # only panel A carries the legend; B is the same encoding
                digits=4, legend=label if first else None)

    colour = {"protocol-driven": "hgred", "structural": "hgblue",
              "unclassified": "hggrey"}
    for ds in m.index:
        anchor, dx, dy = offsets[ds]
        ax.raw(f"\\node[font=\\tiny, text={colour[grp(ds)]}, anchor={anchor}, "
               f"xshift={dx}pt, yshift={dy}pt] at "
               f"(axis cs:{fmt(m.loc[ds, 'values'])},{fmt(m.loc[ds, 'panels_tree'])}) "
               f"{{{nm(ds)}}};")


def fig_dissociation():
    dv, mi = panels("rrstruct"), panels("rrmirror")
    # one placement, shared by both panels, from the development half
    pts = [(float(dv.loc[ds, "values"]), float(dv.loc[ds, "panels_tree"]), nm(ds))
           for ds in dv.index]
    # box: the panel's plot area in points -- 0.44\textwidth by 5.6cm,
    # less the room pgfplots gives the axis labels and ticks
    offsets = dict(zip(dv.index,
                       declutter(pts, (LO, HI), (LO, HI), box=(162, 114), radius=6.5)))
    fig = Fig()
    common = dict(width=r"0.44\textwidth", height="5.6cm",
                  xmin=LO, xmax=HI, ymin=LO, ymax=HI,
                  enlarge_x_limits=False, enlarge_y_limits=False,
                  xtick="0.5,0.6,0.7,0.8,0.9,1.0",
                  ytick="0.5,0.6,0.7,0.8,0.9,1.0",
                  xlabel="predicted from the observed values")

    a = fig.axis("A", "hg axis, hg decimal x=1, hg decimal y=1",
                 legend_style=("font=\\scriptsize, draw=none, fill=none, "
                               "at={(1.17,-0.32)}, anchor=north, legend columns=3, "
                               "column sep=1.2em"),
                 title=r"\textbf{A}\quad fit on the development half",
                 ylabel="predicted from the other panels' presences", **common)
    _scatter_panel(a, dv, True, offsets)

    b = fig.right_of("B", "hg axis, hg decimal x=1, hg decimal y=1", "A", gap="1.5cm",
                     title=r"\textbf{B}\quad fit on the holdout half (mirror)",
                     yticklabels=r"\empty", **common)
    _scatter_panel(b, mi, False, offsets)

    (OUT / "dissociation.tex").write_text(fig.render(__file__))


def fig_maskinc():
    fs = sorted(glob.glob(str(SRC / "results/maskinc/*.csv")))
    if not fs:
        raise SystemExit("no results under results/maskinc/")
    x = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    w = x[x.arm != "mask_full"].groupby(["dataset", "q", "arm"]).auprc.median().unstack("arm")
    gf = (w["values+mask_full"] - w["values"]).unstack("q")
    gk = (w["values+mask_kept"] - w["values"]).unstack("q")
    qs = [0.1, 0.25, 0.5, 0.75, 1.0]

    fig = Fig()
    ax = fig.axis(
        "M", "hg axis, hg decimal y=2",
        width=r"0.60\textwidth", height="5.4cm",
        xlabel="fraction of value columns retained",
        ylabel="gain in average precision",
        xtick=",".join(fmt(q) for q in qs), xmin=0.05, xmax=1.05,
        enlarge_x_limits=False,
        # the top-right corner holds the q=0.1 spike, and the legend is wider
        # than the panel, so it goes underneath in two columns
        legend_style=("font=\\scriptsize, draw=none, fill=none, "
                      "at={(0.5,-0.24)}, anchor=north, legend columns=2, "
                      "column sep=1.4em, row sep=1pt"))
    ax.raw(r"\draw[hg rule] (axis cs:0.05,0) -- (axis cs:1.05,0);")

    # every dataset, faintly, coloured by regime; the medians go on top
    for ds in gf.index:
        c = "hgred" if ds in REG else "hgblue" if ds in STR else "hgpale"
        ax.plot(f"{c}, opacity=0.55, line width=0.6pt, mark=*, mark size=1.1pt, "
                f"mark options={{solid, fill={c}, draw=none}}",
                qs, [gf.loc[ds, q] for q in qs], digits=5, forget=True)
    ax.plot("hgslate, dashed, line width=1.2pt, mark=square*, mark size=2.0pt, "
            "mark options={solid, fill=hgslate, draw=white, line width=0.3pt}",
            qs, [gk[q].median() for q in qs], digits=5,
            legend="retained columns' indicators (median)")
    ax.plot("hgslate, line width=1.2pt, mark=*, mark size=2.0pt, "
            "mark options={fill=hgslate, draw=white, line width=0.3pt}",
            qs, [gf[q].median() for q in qs], digits=5,
            legend="all indicators (median)")
    ax.raw(r"\addlegendimage{hgred, line width=1pt}")
    ax.raw(r"\addlegendentry{one protocol-driven dataset}")
    ax.raw(r"\addlegendimage{hgblue, line width=1pt}")
    ax.raw(r"\addlegendentry{one structural dataset}")

    (OUT / "maskinc.tex").write_text(fig.render(__file__))


if __name__ == "__main__":
    fig_dissociation()
    fig_maskinc()
    print("wrote figures/dissociation.tex and figures/maskinc.tex")
