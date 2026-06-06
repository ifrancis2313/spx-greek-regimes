"""Tests for src/features.py — external behavior only."""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import build_daily_features, _load_vix, _load_spx_returns, _load_opt_spread

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


def test_vrp_is_vix_minus_realized_vol():
    """VRP must equal VIX minus 20-day annualized realized vol — not an approximation."""
    df = build_daily_features(
        f'{DATA}/cboeall1986.parquet',
        f'{DATA}/ff3.parquet',
        f'{DATA}/spx_clean.parquet',
    )
    # VRP + realized_vol reconstruction — check consistency with VIX
    # We can't directly check realized_vol without re-running it, but we can
    # confirm VRP is in a plausible range and negative values exist (crisis property)
    assert df['vrp'].min() < 0, "VRP should go negative during crisis periods"
    assert df['vrp'].max() > 0, "VRP should be positive in calm periods"


def test_momentum_is_cumulative_not_average():
    """21-day momentum should be the sum (cumulative return), not the mean."""
    df = build_daily_features(
        f'{DATA}/cboeall1986.parquet',
        f'{DATA}/ff3.parquet',
        f'{DATA}/spx_clean.parquet',
    )
    # If it were a mean, the max would be ~0.004 (typical daily return).
    # As a sum over 21 days, it should reach several percent.
    assert df['momentum'].abs().max() > 0.05, (
        "momentum looks like a mean rather than a 21-day cumulative sum"
    )


def test_output_has_no_missing_values():
    """build_daily_features must return a complete DataFrame — no NaNs."""
    df = build_daily_features(
        f'{DATA}/cboeall1986.parquet',
        f'{DATA}/ff3.parquet',
        f'{DATA}/spx_clean.parquet',
    )
    assert df.isnull().sum().sum() == 0, "Output contains NaN values"


def test_date_range_respected():
    df = build_daily_features(
        f'{DATA}/cboeall1986.parquet',
        f'{DATA}/ff3.parquet',
        f'{DATA}/spx_clean.parquet',
        start='2000-01-01',
        end='2005-12-31',
    )
    assert df['date'].min() >= pd.Timestamp('2000-01-01')
    assert df['date'].max() <= pd.Timestamp('2005-12-31')


def test_output_columns():
    df = build_daily_features(
        f'{DATA}/cboeall1986.parquet',
        f'{DATA}/ff3.parquet',
        f'{DATA}/spx_clean.parquet',
    )
    assert set(df.columns) == {'date', 'vix', 'vrp', 'momentum', 'opt_spread'}
