"""Tests for src/panel.py — external behavior only."""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from panel import build_panel, GREEKS, MAX_DATE_GAP_DAYS

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


@pytest.fixture(scope='module')
def dummy_regime_labels():
    """Minimal regime label DataFrame covering the options data date range."""
    ff3 = pd.read_parquet(f'{DATA}/ff3.parquet')
    ff3['date'] = pd.to_datetime(ff3['date'])
    dates = ff3[(ff3['date'] >= '1996-01-01') & (ff3['date'] <= '2019-12-31')]['date']
    return pd.DataFrame({'date': dates, 'regime': 'calm'})


@pytest.fixture(scope='module')
def dummy_market_features():
    ff3 = pd.read_parquet(f'{DATA}/ff3.parquet')
    ff3['date'] = pd.to_datetime(ff3['date'])
    ff3 = ff3[(ff3['date'] >= '1996-01-01') & (ff3['date'] <= '2019-12-31')].copy()
    ff3['vix'] = 20.0  # dummy constant
    return ff3[['date', 'mktrf', 'vix']]


@pytest.fixture(scope='module')
def panels(dummy_regime_labels, dummy_market_features):
    return build_panel(
        f'{DATA}/spx_raw.parquet',
        dummy_regime_labels,
        dummy_market_features,
        dte_min=20,
        dte_max=45,
    )


def test_delta_greek_nan_for_first_obs_per_contract(dummy_regime_labels, dummy_market_features):
    """
    The first observation for each contract has no prior day to diff against.
    That row must not appear in the output (it's dropped, not set to NaN).
    """
    puts, calls = build_panel(
        f'{DATA}/spx_raw.parquet',
        dummy_regime_labels,
        dummy_market_features,
        dte_min=20, dte_max=45,
    )
    for df in (puts, calls):
        first_obs = df.groupby('contract_id')['date'].min()
        for contract_id, first_date in first_obs.items():
            contract_rows = df[df['contract_id'] == contract_id]
            assert contract_rows['d_delta'].notna().all(), (
                f"Contract {contract_id} has NaN d_delta"
            )


def test_no_large_date_gaps_in_panel(panels):
    """No consecutive pair within a contract should exceed MAX_DATE_GAP_DAYS calendar days."""
    puts, calls = panels
    for df in (puts, calls):
        df_sorted = df.sort_values(['contract_id', 'date'])
        prev_date = df_sorted.groupby('contract_id')['date'].shift(1)
        gaps = (df_sorted['date'] - prev_date).dt.days.dropna()
        assert gaps.max() <= MAX_DATE_GAP_DAYS, (
            f"Found date gap of {gaps.max()} days — exceeds maximum of {MAX_DATE_GAP_DAYS}"
        )


def test_dte_bounds_respected(panels):
    """All panel observations must fall within the specified DTE window."""
    puts, calls = panels
    for df in (puts, calls):
        assert df['dte'].between(20, 45).all(), "DTE out of [20, 45] range"


def test_puts_and_calls_are_disjoint(panels):
    """Puts and calls panels must not overlap."""
    puts, calls = panels
    assert (puts['cp_flag'] == 'P').all()
    assert (calls['cp_flag'] == 'C').all()


def test_min_contract_obs_enforced(dummy_regime_labels, dummy_market_features):
    """Contracts with fewer than min_contract_obs rows must be excluded."""
    puts, calls = build_panel(
        f'{DATA}/spx_raw.parquet',
        dummy_regime_labels,
        dummy_market_features,
        dte_min=20, dte_max=45,
        min_contract_obs=5,
    )
    for df in (puts, calls):
        counts = df.groupby('contract_id').size()
        assert counts.min() >= 5, f"Contract with fewer than 5 obs found: {counts.min()}"
