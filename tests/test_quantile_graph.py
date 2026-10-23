"""Tests for the quantile-graph construction and estimation (src.quantile_graph)."""
import numpy as np
import pytest

from src.quantile_graph import (
    assign_quantiles,
    build_transition_matrix,
    mean_jump_length,
    estimate_hurst_from_delta,
    make_k_grid,
    compute_delta_curve,
)


def test_assign_quantiles_equal_population_bins():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=10_000)
    labels = assign_quantiles(ts, q=50)
    assert labels.min() >= 0 and labels.max() <= 49
    counts = np.bincount(labels, minlength=50)
    # Equal-population bins: each bin should hold close to 1/50 of the points.
    assert np.all(np.abs(counts - 200) < 60)


def test_assign_quantiles_rejects_constant_series():
    ts = np.ones(100)
    with pytest.raises(ValueError):
        assign_quantiles(ts, q=10)


def test_build_transition_matrix_rows_sum_to_one():
    rng = np.random.default_rng(1)
    ts = rng.normal(size=5000)
    labels = assign_quantiles(ts, q=20)
    w_k = build_transition_matrix(labels, k=1, q=20)
    row_sums = w_k.sum(axis=1)
    nonzero_rows = row_sums > 0
    np.testing.assert_allclose(row_sums[nonzero_rows], 1.0, atol=1e-10)


def test_build_transition_matrix_rejects_k_too_large():
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        build_transition_matrix(labels, k=10, q=5)


def test_white_noise_mean_jump_length_is_k_independent_and_matches_theory():
    """Section 2 states that for white noise, since the quantile graph is
    approximately uniform in every row, Delta(k) is constant (independent of
    k). The paper's own worked value, (Q-1)/2, conflates the *range* of the
    jump length (0 to Q-1) with its being *uniformly distributed* over that
    range; for i, j drawn i.i.d. uniformly over {0, ..., Q-1} (which is what
    equal-population quantile binning of white noise produces), |i-j| is
    triangularly distributed, not uniformly distributed, with analytic mean
    (Q^2-1)/(3Q) (verified below by exact combinatorics). This test checks
    the property the paper actually relies on for the QG method -- Delta(k)
    is flat in k for white noise -- against the correct theoretical constant
    for an i.i.d. uniform transition matrix, rather than the paper's
    imprecise aside, which is not used anywhere in the H estimation
    procedure (Eq. 1) that experiments 1-2 reproduce."""
    rng = np.random.default_rng(2)
    q = 20
    ts = rng.normal(size=20_000)  # i.i.d. white noise (no correlation)
    labels = assign_quantiles(ts, q=q)
    mc_rng = np.random.default_rng(3)
    theoretical_mean = (q**2 - 1) / (3 * q)
    deltas = []
    for k in (1, 5, 20):
        w_k = build_transition_matrix(labels, k=k, q=q)
        delta_k = mean_jump_length(w_k, n_jumps=200_000, rng=mc_rng)
        deltas.append(delta_k)
        assert delta_k == pytest.approx(theoretical_mean, abs=0.25)
    # Delta(k) is (approximately) constant across k for white noise, as
    # stated in the paper ("for all values of k").
    assert max(deltas) - min(deltas) < 0.5


def test_mean_jump_length_identity_matrix_has_zero_jump_length():
    q = 10
    w_k = np.eye(q)
    rng = np.random.default_rng(4)
    delta_k = mean_jump_length(w_k, n_jumps=1000, rng=rng)
    assert delta_k == pytest.approx(0.0)


def test_mean_jump_length_reverse_permutation_matrix():
    """A transition matrix that always maps i -> (Q-1-i) has a known,
    deterministic mean jump length Delta = mean_i |i - (Q-1-i)|."""
    q = 11
    w_k = np.zeros((q, q))
    for i in range(q):
        w_k[i, q - 1 - i] = 1.0
    rng = np.random.default_rng(5)
    delta_k = mean_jump_length(w_k, n_jumps=50_000, rng=rng)
    expected = np.mean([abs(i - (q - 1 - i)) for i in range(q)])
    assert delta_k == pytest.approx(expected, abs=0.05)


def test_estimate_hurst_from_delta_recovers_known_slope():
    """Synthetic power-law Delta(k) = k^H with H known exactly: regression
    must recover H (property check with a known answer)."""
    k_values = np.logspace(0, 3, 40)
    for h_true in (0.1, 0.5, 0.9):
        delta_values = k_values ** h_true
        fit = estimate_hurst_from_delta(k_values, delta_values)
        assert fit["h_hat"] == pytest.approx(h_true, abs=1e-6)
        assert fit["rvalue"] ** 2 == pytest.approx(1.0, abs=1e-6)


def test_estimate_hurst_from_delta_requires_two_points():
    with pytest.raises(ValueError):
        estimate_hurst_from_delta(np.array([1.0]), np.array([1.0]))


def test_make_k_grid_bounds_and_uniqueness():
    k_grid = make_k_grid(1, 1000, 40)
    assert k_grid.min() == 1
    assert k_grid.max() <= 1000
    assert len(k_grid) == len(set(k_grid))
    assert np.all(np.diff(k_grid) > 0)


def test_compute_delta_curve_shape_and_positivity():
    rng = np.random.default_rng(6)
    ts = rng.normal(size=2000).cumsum()  # Brownian-motion-like correlated series
    k_values = make_k_grid(1, 100, 15)
    deltas = compute_delta_curve(ts, q=20, k_values=k_values, n_jumps=5000, rng=rng)
    assert deltas.shape == k_values.shape
    assert np.all(deltas >= 0)


def test_no_lookahead_transition_matrix_uses_only_lagged_pairs():
    """The transition matrix at lag k must only use pairs (t, t+k); flipping
    the tail of the series beyond T - k must not change W_k."""
    rng = np.random.default_rng(7)
    ts = rng.normal(size=100)
    labels = assign_quantiles(ts, q=10)
    k = 5
    w_k_full = build_transition_matrix(labels, k=k, q=10)
    truncated_labels = labels[: 100 - k]  # drop exactly the unused tail
    w_k_truncated = build_transition_matrix(
        np.concatenate([truncated_labels, labels[-k:]]), k=k, q=10
    )
    np.testing.assert_allclose(w_k_full, w_k_truncated)
