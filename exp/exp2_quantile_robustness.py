"""Experiment 2: Robustness of the QG method to the number of quantiles Q.

Reproduces Campanharo & Ramos (2016), Physica A 444, Fig. 5: for
beta = 1.0, 1.2, ..., 3.0 (H = (beta - 1) / 2), the absolute estimation
error |H_hat_QG - H| is computed for Q = 50 and Q = 100 quantiles, showing
that the error stays below 0.02 for all beta regardless of Q (Section 3:
"the absolute difference between QG estimates and exact values of H is
smaller than 0.02 for all beta, and weakly dependent on the choice of Q").

Run: python -m exp.exp2_quantile_robustness [--n-realizations N] [--quick]
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
    T_LENGTH, N_REALIZATIONS, BETA_GRID, Q_GRID_EXP2,
    K_MIN, K_MAX, N_K_POINTS, L_JUMPS, RANDOM_SEED,
)
from src.hurst_qg import run_h_sweep
from data.generate_synthetic import hurst_from_beta

RESULTS_DIR = Path("results")


def main(n_realizations: int = N_REALIZATIONS, t_length: int = T_LENGTH,
         quick: bool = False) -> dict:
    if quick:
        n_realizations = min(n_realizations, 5)
        t_length = min(t_length, 2000)

    k_max = min(K_MAX, t_length - 1)
    h_values = np.array([hurst_from_beta(b, motion=True) for b in BETA_GRID])

    t0 = time.time()
    per_q_records = {}
    for q in Q_GRID_EXP2:
        per_q_records[q] = run_h_sweep(
            h_values=h_values,
            q=q,
            t_length=t_length,
            n_realizations=n_realizations,
            base_seed=RANDOM_SEED + 100_000 * q,
            k_min=K_MIN,
            k_max=k_max,
            n_k_points=N_K_POINTS,
            n_jumps=L_JUMPS,
        )
    elapsed = time.time() - t0

    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for beta, h_exact in zip(BETA_GRID, h_values):
        row = {"beta": float(beta), "h_exact": float(h_exact)}
        for q in Q_GRID_EXP2:
            rec = next(r for r in per_q_records[q] if abs(r["h_exact"] - h_exact) < 1e-9)
            row[f"h_hat_mean_Q{q}"] = rec["h_hat_mean"]
            row[f"abs_error_Q{q}"] = abs(rec["h_hat_mean"] - h_exact)
        rows.append(row)

    max_abs_error = {
        q: max(r[f"abs_error_Q{q}"] for r in rows) for q in Q_GRID_EXP2
    }

    table = {
        "experiment": "exp_2_quantile_robustness",
        "paper": "Campanharo & Ramos (2016), Physica A 444, 43-48",
        "config": {
            "t_length": t_length,
            "q_grid": list(Q_GRID_EXP2),
            "n_realizations": n_realizations,
            "k_min": K_MIN,
            "k_max": k_max,
            "n_k_points": N_K_POINTS,
            "n_jumps": L_JUMPS,
        },
        "runtime_seconds": elapsed,
        "results": rows,
        "max_abs_error_by_q": max_abs_error,
        "expected_outcome": "|H_hat_QG - H| < 0.02 for all beta, for both Q=50 and Q=100",
    }

    with open(RESULTS_DIR / "exp2_quantile_robustness.json", "w") as f:
        json.dump(table, f, indent=2)

    import csv
    with open(RESULTS_DIR / "exp2_quantile_robustness.csv", "w", newline="") as f:
        writer = csv.writer(f)
        row_keys = [c for q in Q_GRID_EXP2 for c in (f"h_hat_mean_Q{q}", f"abs_error_Q{q}")]
        header = ["beta", "H_exact"] + [c for q in Q_GRID_EXP2 for c in (f"H_hat_QG_mean_Q{q}", f"abs_error_Q{q}")]
        writer.writerow(header)
        for row in rows:
            writer.writerow([row["beta"], row["h_exact"]] + [row[k] for k in row_keys])

    # --- Figure 5 analogue: |H_hat_QG - H| vs beta for Q=50, 100 ----------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    betas = [r["beta"] for r in rows]
    for q in Q_GRID_EXP2:
        ax.plot(betas, [r[f"abs_error_Q{q}"] for r in rows], marker="o", label=f"Q={q}")
    ax.axhline(0.02, color="k", linestyle="--", label="0.02 threshold (paper)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$|\hat{H}_{QG} - H|$")
    ax.set_title("QG robustness to the number of quantiles Q")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_abs_error_vs_beta.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: v for k, v in table.items() if k != "results"}, indent=2))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--quick", action="store_true",
                         help="Reduced configuration for a fast smoke run")
    args = parser.parse_args()
    main(n_realizations=args.n_realizations, quick=args.quick)
