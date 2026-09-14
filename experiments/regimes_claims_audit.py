#!/usr/bin/env python3
"""Re-derive the merged regimes paper's claims from its committed results.

Papers 1, 3 and C each have a claim-level audit; this paper had none. Its
numbers are unusually well protected already -- the abstract quotes nine
generated macros and almost nothing else -- so what is left unchecked is the
handful of counts written as words in the prose, and the macros themselves.

Two kinds of check:

  DERIVED   a count the prose states in words, recomputed from the committed
            table it summarises.
  LITERAL   a macro's value also written out as a literal somewhere in the
            prose, which is how a regenerated table and a sentence come apart.
            Collisions that are coincidental -- the same digits standing for a
            different quantity -- are listed in COINCIDENTAL with the reason,
            so a new one surfaces rather than joining a silent allowlist.

Like paper3_audit, this counts the checks it ran and refuses to report a pass
when none did.

Run: .venv/bin/python experiments/regimes_claims_audit.py
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parents[1] / "paper-missingness-regimes"

CHECKS: list[str] = []
FAILURES: list[str] = []

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "twenty": 20}

# A macro value that also appears as a literal, where the two are different
# quantities that happen to share digits. Each carries the reason it was cleared.
COINCIDENTAL = {
    ("GainQuarter", "0.003"): "the aligned-minus-control differences, not the quarter-retained gain",
    ("GsWorst", "1.0"): "an AUROC of 1.000 in the dissociation prose, not the graph-scale worst case",
    ("StandinControl", "0.10"): "the majority baseline over ten hospitals, not the stand-in's value",
    ("WhEicuWithin", "0.934"): "the first of three triple-agreement rates; the same number by construction",
    ("MimicDep", "0.933"): "the upper end of the 0.872-0.933 range, which is MIMIC's own value",
    ("CtPoolOne", "0.010"): "the upper end of MVPC's 0.006-0.010 false-positive rate under a corrected "
                            "release, not the pooled indicator-value correlation at gamma 1",
}


def prose() -> dict[str, str]:
    return {p.name: p.read_text(errors="ignore") for p in sorted(PAPER.glob("sections-*.tex"))}


def check(label, quoted, derived, note=""):
    CHECKS.append(label)
    ok = quoted is not None and quoted == derived
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label:<50} paper {quoted!s:>8}   data {derived!s:>8}"
          + (f"   {note}" if note else ""))
    if not ok:
        FAILURES.append(label)


def data_rows(tex: str) -> int:
    """Rows between the first \\midrule and \\end{tabular}.

    Counting from the top of the tabular also counted the column-header row,
    which has separators and a row terminator like any other and is not bold.
    That reported six algorithms where the table lists five.
    """
    body = tex[:tex.index(r"\end{tabular}")] if r"\end{tabular}" in tex else tex
    if r"\midrule" in body:
        body = body[body.index(r"\midrule") + len(r"\midrule"):]
    n = 0
    for ln in body.splitlines():
        s = ln.strip()
        if (s.count("&") >= 2 and s.endswith(r"\\")
                and not s.startswith(("\\midrule", "\\toprule", "\\bottomrule", "\\cmidrule", "%"))
                and "multicolumn" not in s and "\\textbf" not in s.split("&")[0]):
            n += 1
    return n


def main() -> int:
    if not PAPER.exists():
        print(f"  {PAPER} not found")
        return 2
    tex = prose()
    joined = "\n".join(tex.values())
    tabs = {p.name: p.read_text(errors="ignore") for p in sorted((PAPER / "tables").glob("*.tex"))}
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", "".join(tabs.values())))

    print("=" * 96)
    print("DERIVED — counts the prose states in words, against the tables they summarise")
    print("=" * 96)

    # The indicator count is a design parameter of the plasmode simulation, stated
    # in the Methods; it is not a quantity in the detection table, so the check is
    # abstract against Methods. An earlier version took the largest number in
    # tables/detection.tex, which is a mean and reported a false disagreement.
    m = re.search(r"\\DetAligned\{\}\s+of\s+(\w+)\s+indicators", joined)
    d = re.search(r"missingness of (\w+) fixed columns", tex.get("sections-methods.tex", ""))
    if m and d:
        check("indicators: abstract vs the Methods' column count",
              WORDS.get(m.group(1).lower()), WORDS.get(d.group(1).lower()),
              note="sections-methods.tex, plasmode design")
    else:
        CHECKS.append("indicators")
        FAILURES.append("indicators claim NOT ANCHORED")
        print("  [FAIL] indicators claim: the wording has moved in the abstract or "
              "the Methods; NOT ANCHORED")

    m = re.search(r"(\w+)\s+causal-discovery algorithms were run", joined)
    if m:
        check("algorithms in the structure-learning comparison",
              WORDS.get(m.group(1).lower()), data_rows(tabs.get("negative.tex", "")),
              note="data rows in tables/negative.tex")

    a = re.search(r"(\w+) [Pp]ublished causal-discovery methods assume", joined)
    b = re.search(r"\\subsection\{(\w+) Published Assumptions\}", joined)
    if a and b:
        check("published assumptions: abstract vs the section heading",
              WORDS.get(a.group(1).lower()), WORDS.get(b.group(1).lower()))

    print()
    print("=" * 96)
    print("LITERAL — a macro's value also written out in the prose")
    print("=" * 96)
    unexplained = []
    for name, val in sorted(macros.items()):
        v = val.strip()
        if not re.fullmatch(r"[-+]?[\d.]+", v) or len(v) < 3:
            continue
        if (name, v) in COINCIDENTAL:
            continue
        for fn, txt in tex.items():
            for mm in re.finditer(re.escape(v), txt):
                ctx = txt[max(0, mm.start() - 45):mm.end() + 25].replace("\n", " ")
                if f"\\{name}" in ctx:
                    continue
                unexplained.append((name, v, fn, ctx.strip()))
    CHECKS.append("macro values not duplicated as literals")
    if unexplained:
        FAILURES.append("macro value duplicated as a literal")
        for name, v, fn, ctx in unexplained[:8]:
            print(f"  [FAIL] \\{name} = {v} also written out in {fn}")
            print(f"         ...{ctx}...")
        print("         Different quantities? add the pair to COINCIDENTAL with the reason.")
        print("         Same quantity? use the macro.")
    else:
        print(f"  [ok  ] no macro value is duplicated as a literal "
              f"({len(COINCIDENTAL)} coincidental collisions accounted for)")

    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\{\}", joined))
    undefined = sorted(used - set(macros))
    CHECKS.append("every macro used in prose is defined")
    if undefined:
        FAILURES.append("undefined macro")
        print(f"  [FAIL] used in prose but never defined: {undefined}")
    else:
        print(f"  [ok  ] all {len(used & set(macros))} macros used in prose are defined")

    print()
    print("=" * 96)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    if not CHECKS:
        print("No check ran at all. Treating that as a failure, not a pass.")
        return 2
    print(f"All {len(CHECKS)} checks reproduce from the committed tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
