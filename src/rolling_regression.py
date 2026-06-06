"""
Rolling OLS regressions for Part 1 — Structural Instability.

Demonstrates that Greek sensitivity to market moves is non-stationary:
coefficients spike during stress and collapse during calm, motivating
the formal HMM regime model in Part 2.

Default window: 126 trading days (~6 months). Balances responsiveness
to regime shifts against estimation noise.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f as f_dist


def rolling_ols(
    y: pd.Series,
    X: pd.DataFrame,
    window: int = 126,
) -> pd.DataFrame:
    """
    Estimate OLS over a sliding window.

    Returns a DataFrame indexed by date (last date of each window) with
    columns {var}_coef and {var}_se for each regressor in X, plus 'const'.
    """
    X_c = sm.add_constant(X, has_constant="add")
    records = []

    for end in range(window, len(y) + 1):
        y_w = y.iloc[end - window : end]
        X_w = X_c.iloc[end - window : end]
        try:
            res = sm.OLS(y_w, X_w).fit()
            row: dict = {"date": y.index[end - 1]}
            for var in X_c.columns:
                row[f"{var}_coef"] = res.params[var]
                row[f"{var}_se"] = res.bse[var]
            row["r_squared"] = res.rsquared
            records.append(row)
        except Exception:
            continue

    return pd.DataFrame(records).set_index("date")


def chow_test(
    y: pd.Series,
    X: pd.DataFrame,
    breakpoint_idx: int,
) -> dict:
    """
    Chow test for a structural break at breakpoint_idx.

    H0: coefficients are identical across both sub-samples.
    Returns dict with f_stat, p_value, and the breakpoint index used.
    """
    X_c = sm.add_constant(X, has_constant="add")
    k = X_c.shape[1]
    n = len(y)

    rss_full = sm.OLS(y, X_c).fit().ssr
    rss_1 = sm.OLS(y.iloc[:breakpoint_idx], X_c.iloc[:breakpoint_idx]).fit().ssr
    rss_2 = sm.OLS(y.iloc[breakpoint_idx:], X_c.iloc[breakpoint_idx:]).fit().ssr
    rss_ur = rss_1 + rss_2

    f_stat = ((rss_full - rss_ur) / k) / (rss_ur / (n - 2 * k))
    p_value = float(1 - f_dist.cdf(f_stat, k, n - 2 * k))

    return {
        "f_stat": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "breakpoint_idx": breakpoint_idx,
        "reject_h0": p_value < 0.05,
    }


def rolling_coefficient_summary(
    rolling_df: pd.DataFrame,
    coef_col: str,
    crisis_periods: list[tuple],
) -> pd.DataFrame:
    """
    Summary statistics of a rolling coefficient across known crisis and calm periods.

    crisis_periods: list of (start_date, end_date, label) tuples.
    Returns a DataFrame with mean/std/min/max per period.
    """
    rows = []
    for start, end, label in crisis_periods:
        mask = (rolling_df.index >= start) & (rolling_df.index <= end)
        window = rolling_df.loc[mask, coef_col]
        if len(window) == 0:
            continue
        rows.append(
            {
                "period": label,
                "mean": round(window.mean(), 4),
                "std": round(window.std(), 4),
                "min": round(window.min(), 4),
                "max": round(window.max(), 4),
                "n_days": len(window),
            }
        )
    # Complement: calm (everything not in any crisis)
    crisis_mask = pd.Series(False, index=rolling_df.index)
    for start, end, _ in crisis_periods:
        crisis_mask |= (rolling_df.index >= start) & (rolling_df.index <= end)
    calm = rolling_df.loc[~crisis_mask, coef_col]
    rows.append(
        {
            "period": "calm (all other)",
            "mean": round(calm.mean(), 4),
            "std": round(calm.std(), 4),
            "min": round(calm.min(), 4),
            "max": round(calm.max(), 4),
            "n_days": len(calm),
        }
    )
    return pd.DataFrame(rows)
