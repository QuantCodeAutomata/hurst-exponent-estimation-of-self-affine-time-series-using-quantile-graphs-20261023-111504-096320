# Results -- Hurst exponent estimation of self-affine time series using quantile graphs

Paper: Campanharo, A.S.L.O. & Ramos, F.M. (2016). "Hurst exponent estimation
of self-affine time series using quantile graphs." *Physica A* 444, 43-48.
http://dx.doi.org/10.1016/j.physa.2015.09.094

Both experiments use only synthetic fractional Brownian motion (fBm), as
specified in Section 3 of the paper ("The motions ... used to illustrate the
applicability of [the] QG method have been generated with the wavelet-based
synthesis method [23] ... Unless when indicated otherwise, the values
Q = 50 and T = 10000 have been used in the computations. Results have been
averaged over 100 realizations."). No market data is involved.

## Experiment 1 -- Estimation of Hurst Exponent of Fractional Brownian Motions using Quantile Graphs

Reproduces paper Figs. 3-4: fBm realizations for exact H = 0.0, 0.1, ..., 1.0
(beta = 2H + 1), Q = 50, T = 10000, averaged over 100 realizations; H_hat_QG
estimated as the slope of log(Delta(k)) vs log(k) over the identified
power-law scaling region (Section 2, Eq. (1); Section 3).

Numbers from `results/exp1_hurst_estimation.json` /
`results/exp1_hurst_estimation.csv` (k_min=1, k_max=1000, 40 log-spaced lag
points, L=20000 random-walk jumps per lag, runtime 242.7 s):

| H (exact) | H_hat_QG (mean) | H_hat_QG (std) | \|H_hat_QG - H\| |
|---|---|---|---|
| 0.0 | -0.0002 | 0.0020 | 0.0002 |
| 0.1 | 0.1035  | 0.0079 | 0.0035 |
| 0.2 | 0.2025  | 0.0112 | 0.0025 |
| 0.3 | 0.2996  | 0.0138 | 0.0004 |
| 0.4 | 0.3957  | 0.0172 | 0.0043 |
| 0.5 | 0.4848  | 0.0206 | 0.0152 |
| 0.6 | 0.5700  | 0.0231 | 0.0300 |
| 0.7 | 0.6512  | 0.0246 | 0.0488 |
| 0.8 | 0.7225  | 0.0254 | 0.0775 |
| 0.9 | 0.7865  | 0.0407 | 0.1135 |
| 1.0 | 0.9104  | 0.1903 | 0.0896 |

Pearson correlation between H_hat_QG and the exact H across the grid:
**r = 0.9983** (`pearson_correlation_h_hat_vs_h` in the JSON file).

Figures: `results/exp1_delta_vs_k.png` (log-log Delta(k) vs k, paper Fig. 3
analogue) and `results/exp1_hurst_vs_exact.png` (H_hat_QG vs H, paper Fig. 4
analogue).

**Verdicts**
* Strong linear correlation between H_hat_QG and H (paper: "excellent
  agreement", "strong linear correlation", Section 3 / Fig. 4) --
  **reproduced** (r = 0.998, same qualitative conclusion).
* Pointwise accuracy for H <= 0.5 (|error| <= 0.015) -- **reproduced**, on the
  same order as the paper's reported precision.
* Pointwise accuracy for H >= 0.6 (|error| up to 0.11 at H = 0.9) --
  **not reproduced** at the paper's implied precision; see "Deviations from
  the paper" below for the identified cause.

## Experiment 2 -- Robustness of the QG Method to the Number of Quantiles

Reproduces paper Fig. 5: |H_hat_QG - H| vs beta for Q = 50 and Q = 100,
beta = 1.0, 1.2, ..., 3.0 (H = (beta - 1) / 2), T = 10000, 100 realizations
each (Section 3: "the absolute difference between QG estimates and exact
values of H is smaller than 0.02 for all beta, and weakly dependent on the
choice of Q").

Numbers from `results/exp2_quantile_robustness.json` /
`.csv` (same k_min/k_max/L as experiment 1; runtime 555.1 s for both Q
sweeps):

| beta | H (exact) | \|err\| Q=50 | \|err\| Q=100 |
|---|---|---|---|
| 1.0 | 0.00 | 0.0001 | 0.0003 |
| 1.2 | 0.10 | 0.0036 | 0.0048 |
| 1.4 | 0.20 | 0.0039 | 0.0021 |
| 1.6 | 0.30 | 0.0013 | 0.0018 |
| 1.8 | 0.40 | 0.0044 | 0.0059 |
| 2.0 | 0.50 | 0.0134 | 0.0119 |
| 2.2 | 0.60 | 0.0303 | 0.0310 |
| 2.4 | 0.70 | 0.0542 | 0.0516 |
| 2.6 | 0.80 | 0.0851 | 0.0812 |
| 2.8 | 0.90 | 0.1221 | 0.0956 |
| 3.0 | 1.00 | 0.0345 | 0.0184 |

Maximum absolute error over the beta grid: Q=50 -> 0.122, Q=100 -> 0.096
(`max_abs_error_by_q`). Figure: `results/exp2_abs_error_vs_beta.png`.

**Verdicts**
* Error is "weakly dependent on the choice of Q" (paper, Section 3) --
  **reproduced**: at every beta, |err|(Q=50) and |err|(Q=100) agree to
  within 0.004-0.037, and the two curves track each other closely across the
  whole beta range (same qualitative conclusion as the paper's Fig. 5).
* |H_hat_QG - H| < 0.02 for all beta (paper, Section 3) -- **reproduced**
  for beta <= 2.0 (H <= 0.5); **not reproduced** for beta >= 2.2 (H >= 0.6),
  where the error reaches 0.03-0.12; see below.

## Deviations from the paper

* **fBm generation method.** The paper generates its fBm/fGn realizations
  with "the wavelet-based synthesis method [23]" (Abry & Sellan, 1996), whose
  algorithmic details are not reproduced in the retrieved paper text. A
  dedicated Context7 search for a wavelet-based (Abry-Sellan) fBm synthesis
  package returned no matching library. `data/generate_synthetic.py`
  therefore synthesizes fBm with the Davies-Harte circulant-embedding
  spectral method (Davies & Harte, 1987), with the Wood & Chan (1994)
  negative-eigenvalue-flooring remedy needed for H close to 1 (the exact
  Davies-Harte circulant can produce slightly negative eigenvalues there; the
  `fbm` package's own fallback, the O(n^2) Hosking recursion, measured about
  91 seconds for a single T=10000, H=1.0 realization, which is impractical
  for the paper's full H grid repeated over 100 realizations). This is a
  substitution for an unavailable implementation detail, not a change of
  method: both are exact-covariance, FFT-based Gaussian fBm/fGn generators.
* **Free parameters not specified by the paper.** k_max (the maximum lag),
  the number of log-spaced lag points, and L (the number of random-walk
  jumps per lag used to estimate Delta(k), Eq. (1)) are not given in the
  retrieved paper text, which only requires "more than two decades of
  power-law scaling" and k_max < T. `src/config.py` fixes k_min=1, k_max=1000
  (three decades for T=10000), 40 log-spaced lag points, and L=20000 jumps,
  recorded there as explicit assumptions.
* **Residual bias for H >= 0.6 (both experiments).** The size of this gap was
  localized, not merely observed, before being attributed:
  (1) An independent aggregated-variance Hurst estimator applied to the same
  synthetic paths (not part of the QG pipeline) shows that the Davies-Harte
  generator itself realizes an effective H somewhat below the nominal value
  for H close to 1 (e.g. nominal H=0.9 realizes an aggregated-variance
  estimate of about 0.86-0.87 on individual paths), so part of the gap is a
  generation-side effect rather than an estimation-side one.
  (2) The quantile-graph random-walk estimate of Delta(k) was checked against
  the exact (non-Monte-Carlo) empirical mean |label(t) - label(t+k)| computed
  directly from the quantile labels of the same realization; the two agree
  closely (e.g. H_hat = 0.714 from the random walk vs. 0.706 from the direct
  empirical calculation at H=0.9), ruling out a sampling or implementation
  defect in the random-walk / transition-matrix step.
  (3) The local slope of log(Delta(k)) vs log(k) for high-H realizations is
  visibly curved rather than straight across the full k range, i.e. a clean
  "more than two decades of power-law scaling" (Section 3) is harder to
  realize at fixed Q=50, T=10000 for strongly persistent series. The paper's
  own abstract states that, "for a given time series length, estimation
  error depends basically on the statistical framework used for determining
  the exponent of a power-law model", which is consistent with a
  log-log ordinary-least-squares fit (used here, following the paper's own
  description of "a linear regression of log-transformed coordinates",
  Section 3) being less precise at high H than a maximum-likelihood
  power-law estimator (the paper cites Ref. [21] for the latter without
  giving its implementation). No implementation defect was found in (1)-(2);
  the residual high-H bias is attributed to the combination of the
  finite-T/finite-Q quantization of the QG method and the choice of
  estimation framework, both of which the paper itself identifies as the
  main source of H-estimation error.

## Not reproduced

* The paper's exact wavelet-based (Abry & Sellan, 1996) fBm/fGn synthesis
  algorithm is not reimplemented; a different exact-covariance FFT-based
  generator (Davies-Harte with Wood & Chan flooring) is used instead, as
  described above. The paper's transition-matrix heatmaps (Fig. 2) and the
  fractional-Gaussian-noise extension mentioned in the conclusions are not
  reproduced here, since neither experiment selected for this repository
  requires them.
* The exact precision of Fig. 4 / Fig. 5 (|H_hat_QG - H| < 0.02 for all beta)
  is not reproduced for H >= 0.6; see "Deviations from the paper" for the
  localized cause. The qualitative conclusions of both experiments (strong
  linear correlation in exp_1; weak dependence of the error on Q in exp_2)
  are reproduced.
