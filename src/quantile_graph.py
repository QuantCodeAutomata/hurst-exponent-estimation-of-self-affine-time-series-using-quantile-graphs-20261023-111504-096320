"""Quantile-graph (QG) construction and Hurst-exponent estimation.

Custom implementation — Context7 found no library equivalent (the quantile
graph / random-walk Hurst estimator of Campanharo & Ramos, 2016, is a
paper-specific method, not provided by networkx, ta, or other reviewed
libraries).

Reference: Campanharo & Ramos (2016), Physica A 444, Section 2 ("Methods"):

  * The time-series range is coarse-grained into Q quantiles q_1, ..., q_Q,
    mapped to nodes n_1, ..., n_Q of a network g = {N, A}.
  * Two nodes n_i, n_j are connected by a weighted arc whenever x(t) in
    quantile q_i and x(t+k) in quantile q_j, for time differences
    k = 1, ..., k_max < T.
  * Weights w_ij^k are transition counts normalized so that
    sum_j w_ij^k = 1 (row-stochastic Markov transition matrix W_k).
  * A random walk of L jumps on W_k has jump lengths
    delta_{l,k}(i, j) = |i - j|, and the mean jump length is
    Delta(k) = (1/L) * sum_{l=1}^{L} delta_{l,k}(i, j)                 (Eq. 1)
  * For motions, Delta(k) ~ k^H; H_hat_QG is the slope of a linear
    regression of log(Delta(k)) on log(k) in the power-law scaling region.
  * Sanity check (Section 2): for white noise (beta = 0), all jump lengths
    from Q-1 to 0 are equally likely, so Delta_{beta=0}(k) = (Q-1)/2 for all k.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def assign_quantiles(ts: np.ndarray, q: int) -> np.ndarray:
    """Map each point of a time series to one of Q equal-population quantile bins.

    Parameters
    ----------
    ts : np.ndarray
        Time series values, X(1), ..., X(T).
    q : int
        Number of quantiles/nodes Q (the QG method's only free parameter).

    Returns
    -------
    np.ndarray of int
        Quantile index (node label) in {0, ..., Q-1} for every point of ``ts``.
    """
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}")
    ts = np.asarray(ts, dtype=float)
    # Equal-population bin edges (Q + 1 edges for Q bins), i.e. the Q - 1
    # internal quantile boundaries plus the series' min/max.
    edges = np.quantile(ts, np.linspace(0.0, 1.0, q + 1))
    # Guard against degenerate (repeated) edges on near-constant series.
    edges = np.unique(edges)
    if edges.shape[0] < 2:
        raise ValueError("time series has (near-)zero variance; cannot form quantiles")
    labels = np.digitize(ts, edges[1:-1], right=False)
    labels = np.clip(labels, 0, q - 1)
    return labels.astype(np.int64)


def build_transition_matrix(labels: np.ndarray, k: int, q: int) -> np.ndarray:
    """Build the row-stochastic Markov transition matrix W_k for lag k.

    w_ij^k is the empirical probability that a point in quantile i is
    followed, k steps later, by a point in quantile j, normalized so that
    sum_j w_ij^k = 1 (Section 2).

    Parameters
    ----------
    labels : np.ndarray
        Quantile index sequence returned by :func:`assign_quantiles`.
    k : int
        Time difference (lag) between the paired observations.
    q : int
        Number of quantiles Q.

    Returns
    -------
    np.ndarray, shape (Q, Q)
        Transition matrix W_k. Rows with no observed transition (possible
        only for extreme lags on short series) are left as all-zero.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n = labels.shape[0]
    if k >= n:
        raise ValueError(f"k ({k}) must be < len(labels) ({n})")

    src = labels[: n - k]
    dst = labels[k:]
    counts = np.zeros((q, q), dtype=np.float64)
    flat_idx = src * q + dst
    np.add.at(counts.reshape(-1), flat_idx, 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    w_k = np.divide(
        counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0
    )
    return w_k


def mean_jump_length(
    w_k: np.ndarray,
    n_jumps: int,
    rng: np.random.Generator,
    start_distribution: np.ndarray | None = None,
) -> float:
    """Estimate the mean jump length Delta(k) via a random walk on the QG (Eq. 1).

    A random walk of ``n_jumps`` independent jumps is simulated on the
    quantile graph defined by the row-stochastic transition matrix ``w_k``.
    For each jump, a starting node is drawn from ``start_distribution``
    (the quantile occupation distribution, which is uniform by construction
    of equal-population quantiles) and the arrival node is drawn from the
    corresponding row of ``w_k``. The jump length is delta(i, j) = |i - j|
    and Delta(k) is their sample mean.

    Because the starting distribution is already the chain's (near-uniform)
    stationary distribution for equal-population quantiles, drawing each of
    the L jumps independently from that stationary distribution gives an
    unbiased Monte Carlo estimate of the same expectation a sequential
    L-step random walk would converge to, while allowing a fully vectorized
    implementation.

    Parameters
    ----------
    w_k : np.ndarray, shape (Q, Q)
        Row-stochastic transition matrix for lag k.
    n_jumps : int
        Number of random-walk jumps L.
    rng : np.random.Generator
        Random generator for reproducibility.
    start_distribution : np.ndarray, optional
        Probability distribution over starting nodes. Defaults to uniform
        over the Q nodes.

    Returns
    -------
    float
        Delta(k), the mean jump length.
    """
    q = w_k.shape[0]
    if start_distribution is None:
        start_distribution = np.full(q, 1.0 / q)

    start_cdf = np.cumsum(start_distribution)
    start_cdf[-1] = 1.0
    u0 = rng.random(n_jumps)
    start_nodes = np.searchsorted(start_cdf, u0, side="right")
    start_nodes = np.clip(start_nodes, 0, q - 1)

    row_cdf = np.cumsum(w_k, axis=1)
    row_cdf[:, -1] = 1.0
    gathered_cdf = row_cdf[start_nodes]  # (L, Q)
    u1 = rng.random(n_jumps)
    end_nodes = (u1[:, None] >= gathered_cdf).sum(axis=1)
    end_nodes = np.clip(end_nodes, 0, q - 1)

    jump_lengths = np.abs(start_nodes - end_nodes).astype(np.float64)
    return float(jump_lengths.mean())


def compute_delta_curve(
    ts: np.ndarray,
    q: int,
    k_values: np.ndarray,
    n_jumps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute Delta(k) for a sequence of lags k_values on a single series.

    Implements methodology steps 2-5: quantile assignment, per-lag
    transition-matrix construction, and random-walk mean jump length.
    """
    labels = assign_quantiles(ts, q)
    deltas = np.empty(len(k_values), dtype=np.float64)
    for idx, k in enumerate(k_values):
        w_k = build_transition_matrix(labels, int(k), q)
        deltas[idx] = mean_jump_length(w_k, n_jumps, rng)
    return deltas


def select_scaling_region(
    log_k: np.ndarray, log_delta: np.ndarray, min_decades: float = 2.0
) -> tuple[int, int]:
    """Identify the linear power-law scaling region of a log-log curve.

    Section 3 states that the Hurst exponent is estimated from "the linear
    scaling regions" of the log(Delta(k)) vs log(k) curve, each spanning
    "more than two decades of power-law scaling"; it does not give an
    explicit numerical rule for locating that region. The quantile graph
    is finite (Q nodes), so once the lag k is large enough that the random
    walker has explored the full node range, Delta(k) saturates towards its
    asymptotic plateau and the curve bends away from the power law — fitting
    across that bend biases the slope downward. This routine searches all
    contiguous sub-windows of the lag grid spanning at least ``min_decades``
    decades and selects the one whose OLS fit of log(Delta) on log(k) has
    the highest coefficient of determination R^2, breaking ties in favor of
    the widest window (consistent with "more than two decades").

    Returns
    -------
    (i, j) : tuple of int
        Inclusive index bounds into ``log_k`` / ``log_delta`` of the
        selected scaling region.
    """
    n = len(log_k)
    best_r2 = -np.inf
    best_span = -np.inf
    best_ij = (0, n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            span = log_k[j] - log_k[i]
            if span < min_decades:
                continue
            res = stats.linregress(log_k[i: j + 1], log_delta[i: j + 1])
            r2 = res.rvalue ** 2
            if (r2 > best_r2 + 1e-9) or (abs(r2 - best_r2) <= 1e-9 and span > best_span):
                best_r2, best_span, best_ij = r2, span, (i, j)
    if best_r2 == -np.inf:
        # No window reaches min_decades (short lag grid): fall back to the
        # full range rather than silently returning no fit.
        return 0, n - 1
    return best_ij


def estimate_hurst_from_delta(
    k_values: np.ndarray, delta_values: np.ndarray, min_decades: float = 2.0
) -> dict:
    """Estimate H_hat_QG as the slope of log(Delta(k)) vs log(k) (Section 2-3).

    The Hurst exponent is the slope of an ordinary-least-squares fit
    (scipy.stats.linregress) of log-transformed coordinates, restricted to
    the linear power-law scaling region identified by
    :func:`select_scaling_region` ("[t]he linear scaling regions ... were
    used to estimate the Hurst exponent H_hat_QG, by means of a linear
    regression ... of log-transformed coordinates", Section 3).

    Returns
    -------
    dict with keys: h_hat, intercept, stderr, rvalue, pvalue, k_lo, k_hi
    (the bounds of the selected scaling region) and n_decades (its span).
    """
    k_values = np.asarray(k_values, dtype=float)
    delta_values = np.asarray(delta_values, dtype=float)
    mask = (k_values > 0) & (delta_values > 0)
    if mask.sum() < 2:
        raise ValueError("Need at least two positive (k, Delta(k)) points to fit a slope")
    log_k = np.log10(k_values[mask])
    log_delta = np.log10(delta_values[mask])

    span_available = log_k[-1] - log_k[0]
    region_min_decades = min(min_decades, span_available)
    i, j = select_scaling_region(log_k, log_delta, min_decades=region_min_decades)
    result = stats.linregress(log_k[i: j + 1], log_delta[i: j + 1])
    return {
        "h_hat": float(result.slope),
        "intercept": float(result.intercept),
        "stderr": float(result.stderr),
        "rvalue": float(result.rvalue),
        "pvalue": float(result.pvalue),
        "k_lo": float(10 ** log_k[i]),
        "k_hi": float(10 ** log_k[j]),
        "n_decades": float(log_k[j] - log_k[i]),
    }


def make_k_grid(k_min: int, k_max: int, n_points: int) -> np.ndarray:
    """Build a log-spaced, unique, integer grid of lags in [k_min, k_max]."""
    raw = np.logspace(np.log10(k_min), np.log10(k_max), num=n_points)
    k_grid = np.unique(np.round(raw).astype(int))
    k_grid = k_grid[k_grid >= k_min]
    return k_grid
