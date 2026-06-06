"""Tests for src/rolling_regression.py — external behavior only."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rolling_regression import rolling_ols, chow_test


def make_synthetic(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2000-01-01', periods=n, freq='B')
    X = pd.DataFrame({'x1': rng.normal(0, 1, n), 'x2': rng.normal(0, 1, n)}, index=dates)
    y = pd.Series(0.5 * X['x1'] - 0.3 * X['x2'] + rng.normal(0, 0.1, n), index=dates)
    return y, X


def test_output_indexed_by_date():
    y, X = make_synthetic()
    result = rolling_ols(y, X, window=60)
    assert result.index.dtype == 'datetime64[ns]' or hasattr(result.index, 'date')


def test_output_length():
    """Output should have len(y) - window + 1 rows."""
    y, X = make_synthetic(n=300)
    window = 60
    result = rolling_ols(y, X, window=window)
    assert len(result) == len(y) - window + 1


def test_full_window_converges_to_ols():
    """When window == len(y), rolling OLS should equal full-sample OLS."""
    y, X = make_synthetic(n=200)
    result = rolling_ols(y, X, window=len(y))
    assert len(result) == 1

    full_ols = sm.OLS(y, sm.add_constant(X)).fit()
    rolling_coef = result['x1_coef'].iloc[0]
    full_coef = full_ols.params['x1']
    assert abs(rolling_coef - full_coef) < 1e-8


def test_chow_test_detects_break():
    """Chow test should reject H0 when there is a genuine structural break."""
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range('2000-01-01', periods=n, freq='B')
    X = pd.DataFrame({'x': rng.normal(0, 1, n)}, index=dates)
    # First half: coefficient = 1.0, second half: coefficient = -1.0
    y_vals = np.concatenate([X['x'].values[:100] * 1.0, X['x'].values[100:] * -1.0])
    y_vals += rng.normal(0, 0.05, n)
    y = pd.Series(y_vals, index=dates)

    result = chow_test(y, X, breakpoint_idx=100)
    assert result['reject_h0'], f"Chow test should detect break, got F={result['f_stat']}"


def test_chow_test_no_break():
    """Chow test should fail to reject H0 when coefficients are stable."""
    rng = np.random.default_rng(1)
    n = 200
    dates = pd.date_range('2000-01-01', periods=n, freq='B')
    X = pd.DataFrame({'x': rng.normal(0, 1, n)}, index=dates)
    y = pd.Series(X['x'].values * 0.5 + rng.normal(0, 0.05, n), index=dates)

    result = chow_test(y, X, breakpoint_idx=100)
    assert not result['reject_h0'], f"Chow test should not detect break, got F={result['f_stat']}"
