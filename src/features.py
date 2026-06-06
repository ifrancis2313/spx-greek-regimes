"""
Feature engineering for the HMM regime classifier.

Produces a daily DataFrame with four columns:
  vix          — CBOE VIX closing level (annualized %)
  vrp          — Variance Risk Premium: VIX minus 20-day realized vol (annualized %)
  momentum     — 21-day cumulative SPX return
  opt_spread   — Median options bid-ask spread across all SPX contracts

All features are aligned on trading days in [start, end].
The leading `realized_vol_window` rows are dropped (rolling warm-up).
"""

import numpy as np
import pandas as pd


def build_daily_features(
    vix_path: str,
    ff3_path: str,
    opts_path: str,
    start: str = "1996-01-01",
    end: str = "2019-12-31",
    realized_vol_window: int = 20,
    momentum_window: int = 21,
) -> pd.DataFrame:
    vix_df = _load_vix(vix_path)
    spx_df = _load_spx_returns(ff3_path)
    spread_df = _load_opt_spread(opts_path)

    daily = spx_df.merge(vix_df, on="date", how="inner")
    daily = daily.merge(spread_df, on="date", how="left")
    daily = daily[
        (daily["date"] >= pd.Timestamp(start))
        & (daily["date"] <= pd.Timestamp(end))
    ].sort_values("date").reset_index(drop=True)

    daily["realized_vol"] = (
        daily["spx_ret"].rolling(realized_vol_window).std() * np.sqrt(252) * 100
    )
    daily["vrp"] = daily["vix"] - daily["realized_vol"]
    daily["momentum"] = daily["spx_ret"].rolling(momentum_window).sum()

    return (
        daily[["date", "vix", "vrp", "momentum", "opt_spread"]]
        .dropna()
        .reset_index(drop=True)
    )


def _load_vix(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["Date"])
    return df[["date", "vix"]].dropna().sort_values("date")


def _load_spx_returns(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["spx_ret"] = df["mktrf"] + df["rf"]
    return df[["date", "spx_ret"]].sort_values("date")


def _load_opt_spread(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date")["spread"].median().rename("opt_spread").reset_index()
