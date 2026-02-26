"""Hard-coded numeric targets from the paper for strict replication checks.

Source: Table 2 in "Monetary Policy with Supply Regimes" (Nuño, Renner, Scheidegger, Aug 1, 2025).

All rates are annualized percent (e.g., 2.84 means 2.84%). Output gap is percent.

These targets are intended for automated regression tests. If you change model calibration,
re-train with different seeds, or alter simulation length, small deviations are expected.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Key: (policy, regime)
# regime: "normal" | "bad"
TABLE2_TARGETS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("flex", "normal"): {
        "pi_sss_pct": 0.00, "pi_mean_pct": 0.00, "pi_std_pct": 0.00, "pi_skew": 0.00,
        "x_sss_pct": 0.00, "x_mean_pct": 0.00, "x_std_pct": 0.14, "x_skew": 0.05,
        "r_sss_pct": -0.01, "r_mean_pct": 0.06, "r_std_pct": 0.39, "r_skew": -0.17,
        "i_sss_pct": -0.01, "i_mean_pct": 0.06, "i_std_pct": 0.39, "i_skew": -0.17,
    },
    ("flex", "bad"): {
        "pi_sss_pct": 0.00, "pi_mean_pct": 0.00, "pi_std_pct": 0.00, "pi_skew": 0.00,
        "x_sss_pct": -5.47, "x_mean_pct": -5.46, "x_std_pct": 0.11, "x_skew": -0.11,
        "r_sss_pct": 2.69, "r_mean_pct": 2.74, "r_std_pct": 0.38, "r_skew": -0.35,
        "i_sss_pct": 2.69, "i_mean_pct": 2.74, "i_std_pct": 0.38, "i_skew": -0.35,
    },
    ("commitment", "normal"): {
        "pi_sss_pct": 0.00, "pi_mean_pct": -0.04, "pi_std_pct": 0.24, "pi_skew": -6.12,
        "x_sss_pct": -0.19, "x_mean_pct": -0.08, "x_std_pct": 0.22, "x_skew": -3.29,
        "r_sss_pct": 0.35, "r_mean_pct": 0.64, "r_std_pct": 1.29, "r_skew": 5.28,
        "i_sss_pct": 0.35, "i_mean_pct": 0.60, "i_std_pct": 1.07, "i_skew": 4.92,
    },
    ("commitment", "bad"): {
        "pi_sss_pct": 0.00, "pi_mean_pct": 0.09, "pi_std_pct": 0.35, "pi_skew": 4.61,
        "x_sss_pct": -5.55, "x_mean_pct": -5.32, "x_std_pct": 0.30, "x_skew": 3.86,
        "r_sss_pct": 2.24, "r_mean_pct": 1.80, "r_std_pct": 1.94, "r_skew": -4.42,
        "i_sss_pct": 2.24, "i_mean_pct": 1.89, "i_std_pct": 1.59, "i_skew": -4.31,
    },
    ("discretion", "normal"): {
        "pi_sss_pct": 0.04, "pi_mean_pct": 0.02, "pi_std_pct": 0.07, "pi_skew": -0.79,
        "x_sss_pct": -0.24, "x_mean_pct": -0.11, "x_std_pct": 0.12, "x_skew": -0.07,
        "r_sss_pct": 0.12, "r_mean_pct": 0.21, "r_std_pct": 0.40, "r_skew": -0.21,
        "i_sss_pct": 0.16, "i_mean_pct": 0.23, "i_std_pct": 0.41, "i_skew": -0.23,
    },
    ("discretion", "bad"): {
        "pi_sss_pct": 2.84, "pi_mean_pct": 2.86, "pi_std_pct": 0.07, "pi_skew": 0.74,
        "x_sss_pct": -5.62, "x_mean_pct": -5.46, "x_std_pct": 0.10, "x_skew": 0.05,
        "r_sss_pct": 2.53, "r_mean_pct": 2.54, "r_std_pct": 0.39, "r_skew": -0.42,
        "i_sss_pct": 5.38, "i_mean_pct": 5.40, "i_std_pct": 0.37, "i_skew": -0.42,
    },
}
