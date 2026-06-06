"""Tests for src/regime.py — external behavior only."""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import build_daily_features
from regime import RegimeClassifier

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


@pytest.fixture(scope='module')
def features():
    return build_daily_features(
        f'{DATA}/cboeall1986.parquet',
        f'{DATA}/ff3.parquet',
        f'{DATA}/spx_clean.parquet',
    )


@pytest.fixture(scope='module')
def fitted_classifier(features):
    clf = RegimeClassifier(n_states=4, random_state=42)
    clf.fit(features)
    return clf


def test_predict_returns_valid_labels(fitted_classifier, features):
    """predict() must return integers in [0, n_states-1] for every input row."""
    labels = fitted_classifier.predict(features)
    assert len(labels) == len(features)
    assert set(labels).issubset(set(range(fitted_classifier.n_states)))


def test_reproducibility(features):
    """Same random_state must produce identical state assignments."""
    clf1 = RegimeClassifier(n_states=4, random_state=42).fit(features)
    clf2 = RegimeClassifier(n_states=4, random_state=42).fit(features)
    assert np.array_equal(clf1.predict(features), clf2.predict(features))


def test_high_vix_days_in_same_state(fitted_classifier, features):
    """Days with VIX > 40 should predominantly share a single state (crisis)."""
    labels = fitted_classifier.predict(features)
    high_vix = features['vix'] > 40
    high_vix_labels = labels[high_vix.values]
    if len(high_vix_labels) == 0:
        pytest.skip("No VIX > 40 days in sample")
    dominant_state = pd.Series(high_vix_labels).mode().iloc[0]
    dominant_share = (high_vix_labels == dominant_state).mean()
    assert dominant_share > 0.70, (
        f"High-VIX days split across states (dominant share={dominant_share:.2f})"
    )


def test_label_states_returns_four_names(fitted_classifier, features):
    """label_states() must assign a name to each of the 4 states."""
    names = fitted_classifier.label_states(features)
    assert len(names) == 4
    assert set(names.values()) == {'calm', 'rising_stress', 'recovery', 'crisis'}


def test_transition_matrix_rows_sum_to_one(fitted_classifier, features):
    """Each row of the transition matrix must sum to 1."""
    fitted_classifier.label_states(features)
    mat = fitted_classifier.transition_matrix()
    row_sums = mat.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-6
