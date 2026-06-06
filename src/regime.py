"""
4-state Gaussian Hidden Markov Model for market regime classification.

States are identified by integer labels (0–3) post-training and named
post-hoc by inspecting feature distributions relative to known crisis periods.
Anticipated state mapping (validate empirically):
  lowest VIX mean  → calm
  second           → rising_stress
  third            → recovery
  highest VIX mean → crisis

State names are assigned by VIX mean rank, not hardcoded, so they remain
correct even if the HMM initializes states in a different order across runs.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = ["vix", "vrp", "momentum", "opt_spread"]

# Known crisis validation windows — used to sanity-check state assignments.
KNOWN_CRISES = [
    ("1997-07-01", "1998-10-31", "Asian + LTCM"),
    ("2000-03-01", "2002-10-31", "Dot-com"),
    ("2007-07-01", "2009-06-30", "GFC"),
    ("2011-07-01", "2011-12-31", "Euro debt"),
    ("2015-08-01", "2016-02-29", "China shock"),
    ("2018-10-01", "2018-12-31", "Q4 2018"),
]


class RegimeClassifier:
    def __init__(self, n_states: int = 4, random_state: int = 42, n_iter: int = 1000):
        self.n_states = n_states
        self.random_state = random_state
        self.n_iter = n_iter
        self.model: hmm.GaussianHMM | None = None
        self.scaler = StandardScaler()
        self.state_names: dict[int, str] = {}

    def fit(self, features: pd.DataFrame) -> "RegimeClassifier":
        X = self.scaler.fit_transform(features[FEATURE_COLS].values)
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        X = self.scaler.transform(features[FEATURE_COLS].values)
        return self.model.predict(X)

    def label_states(self, features: pd.DataFrame) -> dict[int, str]:
        """
        Assign interpretable names to integer states by VIX mean rank.
        Lower VIX → calmer regime. Returns mapping {int_state: name}.
        """
        labels = self.predict(features)
        df = features[["vix"]].copy()
        df["state"] = labels
        vix_means = df.groupby("state")["vix"].mean().sort_values()

        # Four-state naming — if n_states != 4, names truncate/extend gracefully
        ordered_names = ["calm", "rising_stress", "recovery", "crisis"]
        self.state_names = {
            state: ordered_names[rank]
            for rank, state in enumerate(vix_means.index)
            if rank < len(ordered_names)
        }
        return self.state_names

    def predict_named(self, features: pd.DataFrame) -> pd.Series:
        """Return regime labels as named strings rather than integers."""
        if not self.state_names:
            self.label_states(features)
        raw = self.predict(features)
        return pd.Series(raw, index=features.index).map(self.state_names)

    def state_summary(self, features: pd.DataFrame) -> pd.DataFrame:
        """Feature distribution statistics per state."""
        labels = self.predict(features)
        df = features[FEATURE_COLS].copy()
        df["state"] = pd.Series(labels, index=features.index).map(self.state_names)
        return df.groupby("state")[FEATURE_COLS].agg(["mean", "std"]).round(4)

    def transition_matrix(self) -> pd.DataFrame:
        """Regime transition probabilities as a named DataFrame."""
        if self.model is None:
            raise RuntimeError("Call fit() before transition_matrix()")
        names = [self.state_names.get(i, str(i)) for i in range(self.n_states)]
        return pd.DataFrame(
            self.model.transmat_,
            index=names,
            columns=names,
        ).round(4)

    def crisis_validation(self, dates: pd.Series, labels: pd.Series) -> pd.DataFrame:
        """
        For each known crisis window, report the dominant regime and its share.
        Used to validate that crisis states align with historical events.
        """
        rows = []
        for start, end, name in KNOWN_CRISES:
            mask = (dates >= start) & (dates <= end)
            window_labels = labels[mask]
            if len(window_labels) == 0:
                continue
            dominant = window_labels.mode().iloc[0]
            share = (window_labels == dominant).mean()
            rows.append(
                {"crisis": name, "dominant_regime": dominant, "regime_share": round(share, 3)}
            )
        return pd.DataFrame(rows)
