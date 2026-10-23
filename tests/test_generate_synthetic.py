"""Tests for synthetic fBm generation (data.generate_synthetic)."""
import numpy as np
import pytest

from data.generate_synthetic import generate_fbm, power_spectrum_beta, hurst_from_beta


def test_generate_fbm_output_length():
    ts = generate_fbm(0.5, 500, seed=1)
    assert ts.shape == (500,)


def test_generate_fbm_first_increment_is_zero_mean():
    """generate_fbm returns X(1), ..., X(T) with the implicit origin
    X(0) = 0 dropped (see its docstring) so the array has exactly
    n_points samples. The defining fBm property B(0) = 0 therefore shows
    up not as ts[0] == 0 pointwise, but as E[X(1)] = E[fgn[0]] = 0: the
    first increment is a zero-mean Gaussian variate, checked here by
    averaging ts[0] over many independent realizations."""
    first_values = np.array([generate_fbm(0.7, 300, seed=s)[0] for s in range(200)])
    assert first_values.mean() == pytest.approx(0.0, abs=0.05)


def test_generate_fbm_reproducible_with_seed():
    ts1 = generate_fbm(0.3, 400, seed=42)
    ts2 = generate_fbm(0.3, 400, seed=42)
    np.testing.assert_allclose(ts1, ts2)


def test_generate_fbm_different_seeds_differ():
    ts1 = generate_fbm(0.3, 400, seed=1)
    ts2 = generate_fbm(0.3, 400, seed=2)
    assert not np.allclose(ts1, ts2)


def test_generate_fbm_invalid_hurst_raises():
    with pytest.raises(ValueError):
        generate_fbm(1.5, 100, seed=1)
    with pytest.raises(ValueError):
        generate_fbm(-0.1, 100, seed=1)


def test_generate_fbm_invalid_length_raises():
    with pytest.raises(ValueError):
        generate_fbm(0.5, 1, seed=1)


def test_brownian_motion_variance_scaling():
    """For H = 0.5, Var[X(t)] should scale approximately linearly with t
    (classical Brownian motion property), averaged over many realizations."""
    rng_seeds = range(30)
    t_length = 2000
    paths = np.vstack([generate_fbm(0.5, t_length, seed=s) for s in rng_seeds])
    var_early = paths[:, 200].var()
    var_late = paths[:, 1800].var()
    ratio = var_late / var_early
    # Expected ratio ~ 1800/200 = 9 for H=0.5 (Var[X(t)] ~ t^{2H} = t)
    assert 4.0 < ratio < 16.0


def test_power_spectrum_beta_motion_relationship():
    assert power_spectrum_beta(0.0, motion=True) == pytest.approx(1.0)
    assert power_spectrum_beta(1.0, motion=True) == pytest.approx(3.0)
    assert power_spectrum_beta(0.5, motion=True) == pytest.approx(2.0)


def test_hurst_from_beta_is_inverse_of_power_spectrum_beta():
    for h in np.arange(0.0, 1.01, 0.1):
        beta = power_spectrum_beta(h, motion=True)
        h_rec = hurst_from_beta(beta, motion=True)
        assert h_rec == pytest.approx(h, abs=1e-9)
