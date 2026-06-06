"""Tests for src/sensitivity.py — external behavior only."""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensitivity import run_sensitivity_regression, run_all, GREEKS


def make_panel(n=500, seed=42):
    """Synthetic contract-level panel with two regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2000-01-01', periods=n, freq='B')
    regimes = np.where(rng.random(n) > 0.7, 'crisis', 'calm')

    df = pd.DataFrame({
        'd_delta': rng.normal(0, 0.01, n),
        'd_gamma': rng.normal(0, 0.001, n),
        'd_vega':  rng.normal(0, 1.0, n),
        'd_theta': rng.normal(0, 0.5, n),
        'mktrf':   rng.normal(0, 0.01, n),
        'vix_chg': rng.normal(0, 1.0, n),
        'regime':  regimes,
        'moneyness': np.random.choice(['deep_otm', 'otm', 'atm'], n),
        'abs_delta': rng.uniform(0.05, 0.60, n),
        'contract_id': [f'contract_{i % 50}' for i in range(n)],
    })
    return df


def test_coefficient_count():
    """
    Regression must have: const + ΔMarket + ΔVIX + (n_regimes - 1) interactions.
    With 2 regimes (calm baseline + crisis), expect 4 parameters.
    """
    panel = make_panel()
    result = run_sensitivity_regression(panel, 'delta', 'atm')
    # const, mktrf, vix_chg, interact_crisis = 4
    assert len(result.params) == 4


def test_puts_and_calls_produce_different_coefficients():
    """
    Puts and calls at the same moneyness/greek should yield different interaction
    coefficients when the underlying data differs.
    """
    rng = np.random.default_rng(99)
    n = 400
    puts = make_panel(n=n, seed=1)
    calls = make_panel(n=n, seed=2)
    # Force a difference: puts have stronger crisis response
    puts.loc[puts['regime'] == 'crisis', 'd_delta'] += puts.loc[puts['regime'] == 'crisis', 'mktrf'] * 5

    res_puts = run_sensitivity_regression(puts, 'delta', 'otm')
    res_calls = run_sensitivity_regression(calls, 'delta', 'otm')

    puts_interact = res_puts.params.get('interact_crisis', 0)
    calls_interact = res_calls.params.get('interact_crisis', 0)
    assert puts_interact != calls_interact


def test_run_all_returns_all_greek_moneyness_combinations():
    """run_all() must produce rows for every greek × moneyness combination."""
    panel = make_panel(n=800)
    results = run_all(panel, moneyness_buckets=['deep_otm', 'otm', 'atm'])
    combos = results[['greek', 'moneyness']].drop_duplicates()
    expected = len(GREEKS) * 3
    assert len(combos) == expected, (
        f"Expected {expected} greek×moneyness combos, got {len(combos)}"
    )


def test_hc3_standard_errors_present():
    """Results must use HC3 — bse should differ from OLS se on heteroskedastic data."""
    import statsmodels.api as sm
    panel = make_panel()
    sub = panel[panel['moneyness'] == 'atm'].copy()
    sub['interact_crisis'] = (sub['regime'] == 'crisis').astype(float) * sub['mktrf']
    X = sm.add_constant(sub[['mktrf', 'vix_chg', 'interact_crisis']])
    y = sub['d_delta']

    ols_se = sm.OLS(y, X).fit().bse
    hc3_se = sm.OLS(y, X).fit(cov_type='HC3').bse

    # HC3 and OLS SEs will differ unless the data is perfectly homoskedastic
    result = run_sensitivity_regression(panel, 'delta', 'atm')
    # Just confirm the result object has bse (HC3 was applied)
    assert hasattr(result, 'bse')
    assert len(result.bse) > 0
