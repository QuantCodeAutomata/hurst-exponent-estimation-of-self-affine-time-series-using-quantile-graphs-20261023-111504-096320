"""Tests for the shared high-level pipeline (src.hurst_qg) used by both
experiments."""
import numpy as np
import pytest

from src.hurst_qg import estimate_hurst_qg, run_h_sweep
from data.generate_synthetic import generate_fbm


def test_estimate_hurst_qg_recovers_known_h_brownian_motion():
    """Standard Brownian motion (H = 0.5) should be recovered within a
    reasonable tolerance from a single long realization."""
    ts = generate_fbm(0.5, 10_000, seed=123)
    rng = np.random.default_rng(123)
    fit = estimate_hurst_qg(ts, q=50, rng=rng, k_min=1, k_max=1000, n_k_points=40, n_jumps=20_000)
    assert 0.3 < fit["h_hat"] < 0.7


def test_estimate_hurst_qg_output_has_expected_keys():
    ts = generate_fbm(0.3, 2000, seed=5)
    rng = np.random.default_rng(5)
    fit = estimate_hurst_qg(ts, q=20, rng=rng, k_min=1, k_max=100, n_k_points=15, n_jumps=2000)
    for key in ("h_hat", "intercept", "stderr", "rvalue", "pvalue", "k_values", "delta"):
        assert key in fit
    assert fit["k_values"].shape == fit["delta"].shape


def test_estimate_hurst_qg_rejects_k_max_too_large():
    ts = generate_fbm(0.5, 100, seed=1)
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        estimate_hurst_qg(ts, q=10, rng=rng, k_min=1, k_max=200, n_k_points=10, n_jumps=100)


def test_run_h_sweep_monotonic_ordering_persisted():
    """H_hat_QG for anti-persistent (H<0.5) vs persistent (H>0.5) fBm should
    be ordered consistently with the generating H (qualitative ranking)."""
    h_values = np.array([0.2, 0.5, 0.8])
    records = run_h_sweep(
        h_values=h_values, q=30, t_length=3000, n_realizations=4,
        base_seed=7, k_min=1, k_max=300, n_k_points=20, n_jumps=5000,
    )
    h_hat_means = [r["h_hat_mean"] for r in records]
    assert h_hat_means[0] < h_hat_means[1] < h_hat_means[2]


def test_run_h_sweep_output_structure():
    h_values = np.array([0.4, 0.6])
    records = run_h_sweep(
        h_values=h_values, q=20, t_length=1500, n_realizations=3,
        base_seed=11, k_min=1, k_max=100, n_k_points=15, n_jumps=2000,
    )
    assert len(records) == 2
    for rec, h in zip(records, h_values):
        assert rec["h_exact"] == pytest.approx(h)
        assert rec["h_hat_values"].shape == (3,)
        assert rec["delta_mean"].shape == rec["k_values"].shape
