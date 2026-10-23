"""High-level QG Hurst-exponent estimation pipeline shared by all experiments.

This module exposes the single public entry point used by every experiment
script in exp/: :func:`estimate_hurst_qg`. Keeping one shared implementation
guarantees that exp_1 and exp_2 apply exactly the same methodology
(quantile assignment -> per-lag transition matrix -> random-walk mean jump
length -> log-log slope), differing only in the parameters (H grid, beta
grid, Q) specified in results/RESULTS.md and src/config.py.
"""
from __future__ import annotations

import numpy as np

from src.quantile_graph import (
    compute_delta_curve,
    estimate_hurst_from_delta,
    make_k_grid,
)
from data.generate_synthetic import generate_fbm
from src.config import K_MIN, K_MAX, N_K_POINTS, L_JUMPS


def estimate_hurst_qg(
    ts: np.ndarray,
    q: int,
    rng: np.random.Generator,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    n_k_points: int = N_K_POINTS,
    n_jumps: int = L_JUMPS,
) -> dict:
    """Estimate the Hurst exponent of a single time series with the QG method.

    Parameters
    ----------
    ts : np.ndarray
        Time series realization (length T).
    q : int
        Number of quantiles/nodes Q (the method's only free parameter).
    rng : np.random.Generator
        Random generator used for the Monte Carlo random walk.
    k_min, k_max, n_k_points : int
        Define the log-spaced lag grid k = k_min, ..., k_max over which the
        scaling region Delta(k) ~ k^H is fit (Section 3: "more than two
        decades of power-law scaling").
    n_jumps : int
        Number of random-walk jumps L used to estimate each Delta(k)
        (Eq. 1).

    Returns
    -------
    dict with keys:
        h_hat : float, estimated Hurst exponent H_hat_QG
        intercept, stderr, rvalue, pvalue : regression diagnostics
        k_values : np.ndarray, the lag grid used
        delta : np.ndarray, Delta(k) for each lag
    """
    if k_max >= len(ts):
        raise ValueError(f"k_max ({k_max}) must be < len(ts) ({len(ts)})")
    k_values = make_k_grid(k_min, k_max, n_k_points)
    deltas = compute_delta_curve(ts, q, k_values, n_jumps, rng)
    fit = estimate_hurst_from_delta(k_values, deltas)
    fit["k_values"] = k_values
    fit["delta"] = deltas
    return fit


def run_h_sweep(
    h_values: np.ndarray,
    q: int,
    t_length: int,
    n_realizations: int,
    base_seed: int,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    n_k_points: int = N_K_POINTS,
    n_jumps: int = L_JUMPS,
) -> list[dict]:
    """Run the full QG estimation pipeline for a grid of exact Hurst values.

    For every H in ``h_values``, generates ``n_realizations`` independent
    fBm paths of length ``t_length``, estimates H_hat_QG for each, and
    returns summary statistics averaged over realizations (Section 3:
    "Results have been averaged over 100 realizations").

    Returns
    -------
    list of dict, one per H value, with keys:
        h_exact, q, h_hat_mean, h_hat_std, h_hat_values (np.ndarray),
        delta_mean (np.ndarray, averaged Delta(k) curve), k_values
    """
    records = []
    for h_idx, h_exact in enumerate(h_values):
        h_hats = np.empty(n_realizations, dtype=np.float64)
        delta_curves = []
        k_values = None
        for real_idx in range(n_realizations):
            seed = base_seed + 10_000 * h_idx + real_idx
            ts = generate_fbm(h_exact, t_length, seed=seed)
            rng = np.random.default_rng(seed)
            fit = estimate_hurst_qg(
                ts, q, rng, k_min=k_min, k_max=k_max,
                n_k_points=n_k_points, n_jumps=n_jumps,
            )
            h_hats[real_idx] = fit["h_hat"]
            delta_curves.append(fit["delta"])
            k_values = fit["k_values"]
        records.append({
            "h_exact": float(h_exact),
            "q": q,
            "h_hat_mean": float(np.mean(h_hats)),
            "h_hat_std": float(np.std(h_hats, ddof=1)),
            "h_hat_values": h_hats,
            "delta_mean": np.mean(np.vstack(delta_curves), axis=0),
            "k_values": k_values,
        })
    return records
