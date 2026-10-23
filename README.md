# Hurst exponent estimation of self-affine time series using quantile graphs

Reproduction of: Campanharo, A.S.L.O. & Ramos, F.M. (2016). "Hurst exponent
estimation of self-affine time series using quantile graphs." *Physica A:
Statistical Mechanics and its Applications*, 444, 43-48.
http://dx.doi.org/10.1016/j.physa.2015.09.094

## What this reproduces

The paper proposes the quantile-graph (QG) method: a time series is coarse-
grained into `Q` quantiles, mapped into the nodes of a weighted directed
graph whose arc weights are the lag-`k` quantile-to-quantile transition
probabilities (a Markov matrix `W_k`), and the Hurst exponent `H` is read off
as the power-law scaling exponent of the mean jump length `Delta(k)` of a
random walker on that graph, `Delta(k) ~ k^H` (Section 2, Eq. (1)).

Two experiments are implemented here:

* **exp_1** -- Estimate `H_hat_QG` for synthetic fractional Brownian motions
  (fBm) with exact `H = 0.0, 0.1, ..., 1.0` (`Q = 50`, `T = 10000`, averaged
  over 100 realizations) and compare against the exact `H` (paper Figs. 3-4).
* **exp_2** -- Robustness of the QG estimate to the number of quantiles `Q`:
  compute `|H_hat_QG - H|` for `Q in {50, 100}` over `beta = 1.0, ..., 3.0`
  (`H = (beta - 1) / 2`) (paper Fig. 5).

No market data is used: both experiments operate entirely on synthetic fBm
realizations, as specified by the paper's own experimental design (Section 3).

## Repository layout

```
src/    quantile-graph construction, transition matrices, random walk,
        power-law fit, and the H sweep used by both experiments
data/   synthetic fBm generation (data/generate_synthetic.py)
exp/    experiment scripts (exp1_hurst_estimation.py, exp2_quantile_robustness.py)
tests/  pytest unit tests for src/ and data/
results/  tables (csv/json), figures (png) and RESULTS.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Full paper configuration (T=10000, 100 realizations each; ~5-10 minutes)
python -m exp.exp1_hurst_estimation
python -m exp.exp2_quantile_robustness

# Reduced smoke configuration (T=2000, 5 realizations; seconds)
python -m exp.exp1_hurst_estimation --quick
python -m exp.exp2_quantile_robustness --quick

# Or run both experiments with the paper configuration:
bash reproduce.sh
```

Tests:

```bash
python -m pytest
```

## Parameters

All paper-derived parameters (`T = 10000`, `Q = 50`, 100 realizations, the
`H`/`beta` grids) live in `src/config.py`, together with the parameters the
paper's methodology section leaves unspecified (`k_max`, the number of
random-walk jumps `L`, the log-spaced lag grid), which are recorded there as
explicit assumptions.

## Results

See `results/RESULTS.md` for the full comparison against the paper, including
a "Deviations from the paper" section (fBm generation method, free
parameters not given in the text) and a "Not reproduced" section.
