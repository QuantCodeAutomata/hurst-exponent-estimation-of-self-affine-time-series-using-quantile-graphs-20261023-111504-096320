#!/usr/bin/env bash
# Reproduce both experiments of Campanharo & Ramos (2016), Physica A 444,
# "Hurst exponent estimation of self-affine time series using quantile
# graphs", with the paper's full configuration (T = 10000, 100 realizations).
set -uo pipefail

cd "$(dirname "$0")"

python3 -m pytest -q || exit 1

python3 -m exp.exp1_hurst_estimation
python3 -m exp.exp2_quantile_robustness

echo "Done. See results/RESULTS.md for the comparison against the paper."
