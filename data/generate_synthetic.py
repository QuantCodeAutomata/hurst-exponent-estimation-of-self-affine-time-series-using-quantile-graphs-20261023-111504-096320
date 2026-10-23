"""Synthetic data generation for the Hurst-exponent / quantile-graph replication.

Both selected experiments (exp_1, exp_2) use only synthetic fractional
Brownian motion (fBm) time series, generated with known exact Hurst exponent
H, as specified in Section 3 of Campanharo & Ramos (2016): "The motions and
noises used to illustrate the applicability of [the] QG method have been
generated with the wavelet-based synthesis method [23] ... Unless when
indicated otherwise, the values Q = 50 and T = 10000 have been used in the
computations." No market data is required by this paper, so no data loader
is included.

This module is the single place where synthetic realizations are produced;
src/ and exp/ call :func:`generate_fbm`, :func:`generate_fbm_realizations`,
:func:`power_spectrum_beta` and :func:`hurst_from_beta` rather than
generating series themselves.

Reference: Campanharo & Ramos (2016), Physica A 444, Section 3 states that
"[t]he motions and noises used to illustrate the applicability of [the] QG
method have been generated with the wavelet-based synthesis method [23]"
(Abry, P., Sellan, F., 1996), with beta = 2H + 1 for motions.

The retrieved paper text cites the wavelet-based synthesis algorithm but does
not reproduce its implementation details (see task uncertainties). Context7
was queried for a dedicated quantitative-finance / signal-processing library
implementing the Abry-Sellan wavelet synthesis and returned no matching
package. This module therefore generates exact-covariance fBm paths with the
Davies-Harte circulant-embedding spectral method (Davies & Harte, 1987), as
explicitly permitted by the experiment implementation notes ("standard
packages like `fbm` or `stochastic` can generate fBm using the Davies-Harte
method"). This substitution for the wavelet-based synthesis method is
recorded as a deviation in results/RESULTS.md.

Custom -- Context7 found no library equivalent that avoids the following
performance defect: the `fbm` package's exact Davies-Harte embedding has
circulant eigenvalues that can go slightly negative for H close to 1 at
moderate n, a known failure mode of the method (Wood, A.T.A., Chan, G.,
"Simulation of stationary Gaussian processes in [0, 1]^d", J. Comput. Graph.
Stat. 3(4), 1994, 409-432 -- cited by the `fbm` package itself). The `fbm`
package's own remedy is to fall back to the O(n^2) Hosking recursion, which
measured ~91 seconds for a single T=10000, H=1.0 realization -- impractical
for the paper's required H=1.0 grid point repeated over many realizations.
Wood & Chan (1994) themselves give the standard alternative: floor the
negative eigenvalues to zero (the "approximate circulant embedding"), which
keeps the method on the O(n log n) FFT path for every H in [0, 1], including
the boundary values H = 0 (fGn reduces to white noise) and H = 1 (fGn becomes
fully correlated, i.e. fBm is a straight line with random slope), both of
which are then well-defined without needing to clip H away from {0, 1}.
"""
from __future__ import annotations

import numpy as np


def _zero_limit_power(base: np.ndarray, exponent: float) -> np.ndarray:
    """Compute base**exponent, resolving the 0**0 indeterminate form by its
    analytic limit (0) rather than NumPy's programming convention (1).

    Needed only at the exact boundary H = 0 (exponent = 2H = 0): for any
    H > 0, 0**(2H) already equals 0 under ordinary floating-point power
    rules, matching lim_{x->0+} 0**x = 0. At H = 0 exactly, NumPy instead
    returns 0**0 = 1, which would make the fGn autocovariance at that
    boundary discontinuous with its H -> 0+ limit; overriding those terms
    to 0 restores continuity (verified numerically against gamma(k) at
    H = 1e-6, 1e-4, 1e-3: see smoke test in tests/test_generate_synthetic.py).
    """
    base = np.asarray(base, dtype=float)
    result = base ** exponent
    if exponent == 0.0:
        result = np.where(base == 0.0, 0.0, result)
    return result


def _fgn_autocovariance(k: np.ndarray, hurst: float) -> np.ndarray:
    """Theoretical autocovariance of unit-variance fractional Gaussian noise.

    gamma(k) = 0.5 * (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H}), the standard fGn
    autocovariance (Davies & Harte, 1987, Eq. 2.1; also used internally by
    the `fbm` package's `_autocovariance`).
    """
    k = np.asarray(k, dtype=float)
    two_h = 2.0 * hurst
    return 0.5 * (
        _zero_limit_power(np.abs(k - 1), two_h)
        - 2.0 * _zero_limit_power(np.abs(k), two_h)
        + _zero_limit_power(np.abs(k + 1), two_h)
    )


def _davies_harte_fgn(hurst: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample unit-variance fractional Gaussian noise of length ``n``.

    Implements the Davies-Harte circulant-embedding spectral method (Davies &
    Harte, 1987) with the Wood & Chan (1994) eigenvalue-flooring remedy for
    the near-H=1 failure mode, following the same circulant-matrix
    construction as the `fbm` package's `_daviesharte` (first row of the
    2n x 2n circulant matrix, eigenvalues via FFT), but clipping any
    negative eigenvalues to zero instead of raising/falling back to the
    O(n^2) Hosking recursion.
    """
    row_component = _fgn_autocovariance(np.arange(1, n), hurst)
    row = np.concatenate((
        [_fgn_autocovariance(np.array([0.0]), hurst)[0]],
        row_component,
        [0.0],
        row_component[::-1],
    ))
    m = 2 * n
    assert row.shape[0] == m

    eigenvals = np.fft.fft(row).real
    eigenvals = np.clip(eigenvals, 0.0, None)  # Wood & Chan (1994) flooring

    gn = rng.standard_normal(n)
    gn2 = rng.standard_normal(n)

    w = np.zeros(m, dtype=complex)
    w[0] = np.sqrt(eigenvals[0] / m) * gn[0]
    idx = np.arange(1, n)
    w[idx] = np.sqrt(eigenvals[idx] / (2.0 * m)) * (gn[idx] + 1j * gn2[idx])
    w[n] = np.sqrt(eigenvals[n] / m) * gn2[0]
    idx2 = np.arange(n + 1, m)
    w[idx2] = np.sqrt(eigenvals[idx2] / (2.0 * m)) * (gn[m - idx2] - 1j * gn2[m - idx2])

    z = np.fft.fft(w)
    fgn = z[:n].real
    return fgn


def generate_fbm(hurst: float, n_points: int, seed: int | None = None) -> np.ndarray:
    """Generate a single fractional Brownian motion realization.

    Parameters
    ----------
    hurst : float
        Hurst exponent H in [0, 1]. H = 0.5 corresponds to standard
        Brownian motion.
    n_points : int
        Number of points T in the output series (the paper's T = 10000).
    seed : int, optional
        Seed for the NumPy random generator, for reproducibility across
        realizations.

    Returns
    -------
    np.ndarray
        Array of length ``n_points`` containing one fBm realization,
        X(1), ..., X(T), with X(0) = 0 dropped so the path has exactly
        ``n_points`` samples.
    """
    if not (0.0 <= hurst <= 1.0):
        raise ValueError(f"hurst must be in [0, 1], got {hurst}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    rng = np.random.default_rng(seed)

    n = n_points  # number of increments; path = cumsum(increments)
    fgn = _davies_harte_fgn(hurst, n, rng)
    # Scale from the unit-covariance process to the unit-interval [0, length]
    # fBm, following the same (length / n) ** hurst scaling used by the `fbm`
    # package's `FBM.fgn` (length = 1.0 here).
    fgn_scaled = fgn * (1.0 / n) ** hurst
    path = np.cumsum(fgn_scaled)
    assert path.shape[0] == n_points
    return path


def power_spectrum_beta(hurst: float, motion: bool = True) -> float:
    """Return the power-spectrum exponent beta associated with a Hurst value.

    For motions: beta = 2H + 1 (Section 1, Section 3).
    For noises:  beta = 2H - 1 (Section 3).
    """
    return 2.0 * hurst + 1.0 if motion else 2.0 * hurst - 1.0


def hurst_from_beta(beta: float, motion: bool = True) -> float:
    """Inverse of :func:`power_spectrum_beta` for motions (H = (beta - 1) / 2)."""
    return (beta - 1.0) / 2.0 if motion else (beta + 1.0) / 2.0


def generate_fbm_realizations(
    hurst: float, t_length: int, n_realizations: int, base_seed: int
) -> list[np.ndarray]:
    """Generate ``n_realizations`` independent fBm paths for a given H.

    Parameters
    ----------
    hurst : float
        Exact Hurst exponent used to generate every realization.
    t_length : int
        Length T of each realization (paper default T = 10000).
    n_realizations : int
        Number of independent realizations (paper default 100).
    base_seed : int
        Base seed; realization i uses seed ``base_seed + i`` for
        reproducibility.

    Returns
    -------
    list of np.ndarray
        ``n_realizations`` fBm paths, each of length ``t_length``.
    """
    if t_length < 2:
        raise ValueError(f"t_length must be >= 2, got {t_length}")
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}")
    return [
        generate_fbm(hurst, t_length, seed=base_seed + i)
        for i in range(n_realizations)
    ]
