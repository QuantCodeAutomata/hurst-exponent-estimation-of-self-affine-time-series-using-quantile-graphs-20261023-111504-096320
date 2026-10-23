"""Experiment 1: Estimation of the Hurst exponent of fBm using quantile graphs.

Reproduces Campanharo & Ramos (2016), Physica A 444, Figs. 3-4: synthetic
fractional Brownian motions are generated for exact Hurst exponents
H = 0.0, 0.1, ..., 1.0 (beta = 2H + 1), mapped into quantile graphs with
Q = 50 quantiles, and H_hat_QG is estimated as the slope of
log(Delta(k)) vs log(k) in the power-law scaling region. H_hat_QG is then
compared against the exact H values used to generate the series.

Run: python -m exp.exp1_hurst_estimation [--n-realizations N] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    T_LENGTH, Q_DEFAULT, N_REALIZATIONS, H_GRID,
    K_MIN, K_MAX, N_K_POINTS, L_JUMPS, RANDOM_SEED,
)
from src.hurst_qg import run_h_sweep

RESULTS_DIR = Path("results")


def main(n_realizations: int = N_REALIZATIONS, t_length: int = T_LENGTH,
         quick: bool = False) -> dict:
    if quick:
        n_realizations = min(n_realizations, 5)
        t_length = min(t_length, 2000)

    k_max = min(K_MAX, t_length - 1)

    t0 = time.time()
    records = run_h_sweep(
        h_values=H_GRID,
        q=Q_DEFAULT,
        t_length=t_length,
        n_realizations=n_realizations,
        base_seed=RANDOM_SEED,
        k_min=K_MIN,
        k_max=k_max,
        n_k_points=N_K_POINTS,
        n_jumps=L_JUMPS,
    )
    elapsed = time.time() - t0

    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Machine-readable results ----------------------------------------
    table = {
        "experiment": "exp_1_hurst_estimation",
        "paper": "Campanharo & Ramos (2016), Physica A 444, 43-48",
        "config": {
            "t_length": t_length,
            "q": Q_DEFAULT,
            "n_realizations": n_realizations,
            "k_min": K_MIN,
            "k_max": k_max,
            "n_k_points": N_K_POINTS,
            "n_jumps": L_JUMPS,
        },
        "runtime_seconds": elapsed,
        "results": [
            {
                "h_exact": r["h_exact"],
                "h_hat_mean": r["h_hat_mean"],
                "h_hat_std": r["h_hat_std"],
                "abs_error": abs(r["h_hat_mean"] - r["h_exact"]),
            }
            for r in records
        ],
    }
    h_exact = np.array([r["h_exact"] for r in table["results"]])
    h_hat = np.array([r["h_hat_mean"] for r in table["results"]])
    corr = float(np.corrcoef(h_exact, h_hat)[0, 1])
    table["pearson_correlation_h_hat_vs_h"] = corr

    with open(RESULTS_DIR / "exp1_hurst_estimation.json", "w") as f:
        json.dump(table, f, indent=2)

    import csv
    with open(RESULTS_DIR / "exp1_hurst_estimation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["H_exact", "H_hat_QG_mean", "H_hat_QG_std", "abs_error"])
        for row in table["results"]:
            writer.writerow([row["h_exact"], row["h_hat_mean"], row["h_hat_std"], row["abs_error"]])

    # --- Figure 4 analogue: H_hat_QG vs H ---------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(h_exact, h_hat, yerr=[r["h_hat_std"] for r in records],
                fmt="o", capsize=3, label=r"$\hat{H}_{QG}$ (QG estimate)")
    ax.plot([0, 1], [0, 1], "k--", label="identity (exact H)")
    ax.set_xlabel("Exact Hurst exponent H")
    ax.set_ylabel(r"Estimated $\hat{H}_{QG}$")
    ax.set_title(f"QG Hurst estimation of fBm (Q={Q_DEFAULT}, T={t_length})")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_hurst_vs_exact.png", dpi=150)
    plt.close(fig)

    # --- Figure 3 analogue: log-log Delta(k) vs k for each beta -----------
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in records:
        ax.plot(r["k_values"], r["delta_mean"], marker=".",
                label=f"H={r['h_exact']:.1f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\Delta(k)$")
    ax.set_title("Mean jump length vs lag (log-log scaling)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_delta_vs_k.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: v for k, v in table.items() if k != "results"}, indent=2))
    print(f"Pearson correlation(H_hat_QG, H) = {corr:.5f}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--quick", action="store_true",
                         help="Reduced configuration for a fast smoke run")
    args = parser.parse_args()
    main(n_realizations=args.n_realizations, quick=args.quick)
