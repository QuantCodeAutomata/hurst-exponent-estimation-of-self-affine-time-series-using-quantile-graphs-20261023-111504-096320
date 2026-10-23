"""Paper-derived parameters for the quantile-graph (QG) Hurst estimator.

Reference: Campanharo, A.S.L.O. & Ramos, F.M. (2016). "Hurst exponent
estimation of self-affine time series using quantile graphs." Physica A 444,
43-48.

Section 3 ("Results") states: "Unless when indicated otherwise, the values
Q = 50 and T = 10000 have been used in the computations. Results have been
averaged over 100 realizations." fBm series were generated with
beta = 2H + 1 for motions (Section 1 and Section 3).

Experiment 2 (Fig. 5) sweeps Q in {50, 100} against beta and reports that
|H_hat_QG - H| < 0.02 for all beta, independent of Q.

Parameters not stated explicitly in the retrieved paper text (k_max, the
number of random-walk jumps L, and the log-spacing of the k grid) are
recorded here as explicit assumptions rather than silently hard-coded inside
the estimation routines, see results/RESULTS.md ("Deviations from the
paper").
"""
from __future__ import annotations

import numpy as np

# --- Confirmed by the paper (Section 3) ---------------------------------
T_LENGTH: int = 10_000          # length of each synthetic fBm realization
Q_DEFAULT: int = 50             # default number of quantiles/nodes
N_REALIZATIONS: int = 100       # number of independent realizations averaged

# H grid for experiment 1: H = 0.0, 0.1, ..., 1.0
H_GRID: np.ndarray = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)

# beta grid for experiment 2: beta = 1.0, 1.2, ..., 3.0 -> H = (beta - 1) / 2
BETA_GRID: np.ndarray = np.round(np.arange(1.0, 3.0 + 1e-9, 0.2), 1)
Q_GRID_EXP2 = (50, 100)

# --- Assumptions (not specified by the retrieved paper text) -------------
# k_max must satisfy k_max < T and must display "more than two decades" of
# power-law scaling in the log-log Delta(k) vs k plot (Section 3, Fig. 3).
# k_max = 1000 with k_min = 1 spans three decades (10^0 to 10^3) for
# T = 10000, comfortably exceeding the two-decade requirement while keeping
# k_max well below T.
K_MIN: int = 1
K_MAX: int = 1_000
N_K_POINTS: int = 40  # number of log-spaced lag values used to fit the slope

# Number of jumps L of the random walk used to estimate the mean jump length
# Delta(k) = (1/L) * sum_l delta_{l,k}(i,j), Eq. (1). The paper does not
# state L; a large value is chosen so that the Monte Carlo standard error of
# Delta(k) is negligible compared to the scaling signal being fit.
L_JUMPS: int = 20_000

RANDOM_SEED: int = 12345
