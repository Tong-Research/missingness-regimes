#!/usr/bin/env python3
"""Bank a long run's results unit by unit, and resume from what is on disk.

Why this exists. `probe_sweepnull.py` wrote its CSV only after the last of ten cells, so an
eight-hour wall clock destroyed nine finished cells of `cand_acs_income-p4`; `cand_eicu_regime`
needed about 54 hours a seed under contention and could never have finished at all.
`run_candidate.py` had a subtler version: it rewrote the WHOLE accumulated file after every unit,
so a kill inside that window lost the banked units too, and its resume rule could not tell
"complete" from "all partial" when only one unit was on file. Both were fixed by hand, separately,
with the same ideas. This is those ideas in one place, so the next long-running probe gets them
for free rather than rediscovering them after losing a night.

The four properties that matter, each learned from a real loss:

1. **Append per unit, then flush and fsync.** A unit that finished must survive the next kill.
2. **A unit counts as done only when every member is present.** A cell with one arm written and
   the other not is half-measured, and resuming from it silently reports a partial cell as real.
3. **Repair through a temp file and an atomic replace.** Rewriting the target in place puts it in
   a truncated state, and a kill in that window destroys everything banked -- the exact failure
   the checkpointing exists to prevent.
4. **Never trust a lone unit.** With a single unit on file there is nothing to compare against, so
   "as many rows as the fullest unit" calls a half-written unit complete. One unit is cheap to
   recompute; ambiguity is not.

Usage:

    ARMS = ("mean_impute", "family_cv")
    with BankedWriter(out, FIELDS, key=("seed", "fold"), member="arm", members=ARMS) as w:
        done = w.completed()
        for seed, fold in cells:
            if (seed, fold) in done:
                continue
            w.write_unit([row_for(a) for a in ARMS])
        return w.exit_code(expected=len(cells))
"""
from __future__ import annotations

import csv
import os
import pathlib


class BankedWriter:
    def __init__(self, path, fieldnames, key, member, members, cast=str, verbose=True):
        self.path = pathlib.Path(path)
        self.fieldnames = list(fieldnames)
        self.key = tuple(key)
        self.member = member
        self.members = tuple(members)
        # CSV gives every field back as a string, while callers loop over the native type they
        # wrote (ints, usually). Without a cast, `if (seed, fold) in done` is False for every
        # already-banked unit and the whole point of resuming is quietly lost.
        self.cast = cast
        self.verbose = verbose
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done, self._kept = self._scan()
        self._new = 0
        self._fh = None
        self._w = None

    def _keyof(self, row):
        return tuple(self.cast(row[k]) for k in self.key)

    def _scan(self):
        """Units already complete on disk, and the rows backing them.

        Rows that fail to parse are dropped: a job killed mid-write can leave a torn final line.
        Units missing any member are dropped whole, per property 2.
        """
        if not self.path.exists():
            return set(), []
        by_unit: dict[tuple, dict] = {}
        with self.path.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    k = self._keyof(row)
                    m = row[self.member]
                except (KeyError, TypeError):
                    continue                                  # torn or truncated line
                if any(v is None for v in k) or m is None:
                    continue
                by_unit.setdefault(k, {})[m] = row
        done, kept = set(), []
        for k, got in sorted(by_unit.items()):
            if all(m in got for m in self.members):
                done.add(k)
                kept.extend(got[m] for m in self.members)
        # Property 4: one unit on file proves nothing about its own completeness.
        if len(by_unit) <= 1 and len(done) <= 1:
            return set(), []
        return done, kept

    def completed(self):
        """The set of unit keys already banked, each a tuple of the `key` columns cast by `cast`."""
        return set(self._done)

    def __enter__(self):
        # Property 3: repair through a temp file and rename, never by truncating the target.
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()
            w.writerows(self._kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self._fh = self.path.open("a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if self.verbose and self._done:
            print(f"  resuming: {len(self._done)} unit(s) already banked in {self.path.name}",
                  flush=True)
        return self

    def write_unit(self, rows):
        """Append one complete unit and make it durable before the next one starts."""
        rows = list(rows)
        if len(rows) != len(self.members):
            raise ValueError(f"a unit must carry {len(self.members)} rows, got {len(rows)}")
        self._w.writerows(rows)
        self._fh.flush()
        os.fsync(self._fh.fileno())          # property 1
        self._new += 1

    def __exit__(self, *exc):
        if self._fh:
            self._fh.close()
        return False

    @property
    def banked(self):
        return len(self._done) + self._new

    def exit_code(self, expected):
        """0 when every expected unit is banked, 2 when the run is partial, 1 when nothing is.

        A partial run is a normal intermediate state, not a success: exiting non-zero keeps the
        farm from filing it as done, and keeps a scorer from averaging half a design.
        """
        if self.banked == 0:
            return 1
        return 0 if self.banked >= expected else 2
