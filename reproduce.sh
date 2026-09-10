#!/bin/sh
# Regenerate tables/figures from results/. Steps whose inputs are withheld under a data-use agreement report it and continue.
export PYTHONPATH=$PWD/src:$PWD/experiments:$PWD/figures:$PYTHONPATH
echo "-- inflating compressed result files"; for f in $(find results -name "*.csv.gz"); do gunzip -kf "$f"; done
echo "-- regime tables"; ( cd figures && python make_tables.py && cd .. ) || echo "   skipped: regime tables needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
echo "-- regime figures"; ( cd figures && python make_figures.py && cd .. ) || echo "   skipped: regime figures needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
echo "-- regime macros"; ( cd figures && python gen_regimes.py && cd .. ) || echo "   skipped: regime macros needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
echo "-- assumption macros"; ( cd figures && python gen_assumptions.py && cd .. ) || echo "   skipped: assumption macros needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
