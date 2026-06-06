"""
Contract-level panel builder for Part 3 — Greek Sensitivity Analysis.

Constructs a (contract, day) panel from spx_raw.parquet where each row
is a consecutive-day observation of a specific option contract. Day-over-day
ΔGreek values serve as the dependent variables in the sensitivity regressions.

Contract identity: (secid, strike_price, exdate, cp_flag)
Primary sample:    20–45 DTE
FOMC window:       5–20 DTE (robustness check)
"""

import numpy as np
import pandas as pd


GREEKS = ["delta", "gamma", "vega", "theta"]
MONEYNESS_BINS = [0.0, 0.20, 0.40, 0.60, 1.01]
MONEYNESS_LABELS = ["deep_otm", "otm", "atm", "itm"]

# Maximum calendar-day gap between consecutive observations within a contract.
# Covers weekends (2 days) and single holidays (3 days). Gaps > 5 signal
# a missing trading day and are dropped to avoid spurious large ΔGreeks.
MAX_DATE_GAP_DAYS = 5


def build_panel(
    raw_path: str,
    regime_labels: pd.DataFrame,
    market_features: pd.DataFrame,
    dte_min: int = 20,
    dte_max: int = 45,
    min_contract_obs: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the contract-level panel and split into puts and calls.

    Parameters
    ----------
    raw_path : str
        Path to spx_raw.parquet.
    regime_labels : pd.DataFrame
        Must contain columns ['date', 'regime'] from RegimeClassifier.predict_named().
    market_features : pd.DataFrame
        Must contain columns ['date', 'mktrf', 'vix'] for regression covariates.
    dte_min, dte_max : int
        DTE filter bounds (inclusive).
    min_contract_obs : int
        Minimum consecutive observations required to keep a contract.

    Returns
    -------
    puts : pd.DataFrame
    calls : pd.DataFrame
    """
    df = _load_and_filter(raw_path, dte_min, dte_max)
    df = _compute_delta_greeks(df)
    df = _add_moneyness(df)
    df = _merge_covariates(df, regime_labels, market_features)
    df = _drop_short_contracts(df, min_contract_obs)

    puts = df[df["cp_flag"] == "P"].copy().reset_index(drop=True)
    calls = df[df["cp_flag"] == "C"].copy().reset_index(drop=True)
    return puts, calls


def _load_and_filter(path: str, dte_min: int, dte_max: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])
    df["dte"] = (df["exdate"] - df["date"]).dt.days

    df = df[(df["dte"] >= dte_min) & (df["dte"] <= dte_max)]
    df = df[(df[GREEKS] != 0).all(axis=1)]

    df["contract_id"] = (
        df["secid"].astype(str)
        + "_"
        + df["strike_price"].astype(str)
        + "_"
        + df["exdate"].astype(str)
        + "_"
        + df["cp_flag"]
    )
    return df.sort_values(["contract_id", "date"]).reset_index(drop=True)


def _compute_delta_greeks(df: pd.DataFrame) -> pd.DataFrame:
    for greek in GREEKS:
        df[f"d_{greek}"] = df.groupby("contract_id")[greek].diff()

    # Drop first observation per contract (no prior day to diff against)
    df = df.dropna(subset=[f"d_{g}" for g in GREEKS])

    # Drop observations with date gaps > MAX_DATE_GAP_DAYS (non-consecutive days)
    df["prev_date"] = df.groupby("contract_id")["date"].shift(1)
    df["date_gap"] = (df["date"] - df["prev_date"]).dt.days
    df = df[df["date_gap"] <= MAX_DATE_GAP_DAYS].drop(
        columns=["prev_date", "date_gap"]
    )
    return df


def _add_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_delta"] = df["delta"].abs()
    df["moneyness"] = pd.cut(
        df["abs_delta"],
        bins=MONEYNESS_BINS,
        labels=MONEYNESS_LABELS,
        right=True,
    )
    return df


def _merge_covariates(
    df: pd.DataFrame,
    regime_labels: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    df = df.merge(regime_labels[["date", "regime"]], on="date", how="left")
    df = df.merge(market_features[["date", "mktrf", "vix"]], on="date", how="left")
    df["vix_chg"] = df.groupby("date")["vix"].transform("first").diff()
    return df.dropna(subset=["regime", "mktrf", "vix_chg"])


def _drop_short_contracts(df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    counts = df.groupby("contract_id").size()
    valid = counts[counts >= min_obs].index
    return df[df["contract_id"].isin(valid)]
