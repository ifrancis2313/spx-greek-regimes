"""
Regime-interacted Greek sensitivity regressions for Part 3.

Model (per Greek, moneyness bucket, instrument type):

  d_greek = α + β1·ΔMarket + β2·ΔVix
          + γ2·(Regime2 × ΔMarket)
          + γ3·(Regime3 × ΔMarket)
          + γ4·(Regime4 × ΔMarket)
          + ε

Baseline regime (calm) is the omitted category.
γk coefficients measure additional Greek sensitivity relative to calm.
HC3-robust standard errors throughout (expected heteroskedasticity from vol clustering).
"""

import pandas as pd
import statsmodels.api as sm


GREEKS = ["delta", "gamma", "vega", "theta"]
MONEYNESS_BUCKETS = ["deep_otm", "otm", "atm"]
BASELINE_REGIME = "calm"


def run_sensitivity_regression(
    panel: pd.DataFrame,
    greek: str,
    moneyness: str,
    market_col: str = "mktrf",
    vix_chg_col: str = "vix_chg",
    regime_col: str = "regime",
    baseline_regime: str = BASELINE_REGIME,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Run the regime-interacted sensitivity regression for one Greek × moneyness cell.

    Interaction terms are (regime_dummy × ΔMarket) for each non-baseline regime.
    Returns a fitted statsmodels results object with HC3 standard errors.
    """
    sub = (
        panel[panel["moneyness"] == moneyness]
        .copy()
        .dropna(subset=[f"d_{greek}", market_col, vix_chg_col, regime_col])
    )

    non_baseline = sorted(r for r in sub[regime_col].unique() if r != baseline_regime)

    for regime in non_baseline:
        sub[f"interact_{regime}"] = (
            (sub[regime_col] == regime).astype(float) * sub[market_col]
        )

    regressors = [market_col, vix_chg_col] + [f"interact_{r}" for r in non_baseline]
    X = sm.add_constant(sub[regressors], has_constant="add")
    y = sub[f"d_{greek}"]

    return sm.OLS(y, X).fit(cov_type="HC3")


def run_all(
    panel: pd.DataFrame,
    market_col: str = "mktrf",
    vix_chg_col: str = "vix_chg",
    moneyness_buckets: list[str] = MONEYNESS_BUCKETS,
) -> pd.DataFrame:
    """
    Run all Greek × moneyness sensitivity regressions and return a tidy results table.

    Each row is one (greek, moneyness, parameter) combination with its coefficient,
    standard error, t-statistic, p-value, and observation count.
    """
    records = []
    for greek in GREEKS:
        for bucket in moneyness_buckets:
            try:
                result = run_sensitivity_regression(
                    panel, greek, bucket, market_col, vix_chg_col
                )
                for param in result.params.index:
                    records.append(
                        {
                            "greek": greek,
                            "moneyness": bucket,
                            "parameter": param,
                            "coef": result.params[param],
                            "se": result.bse[param],
                            "t_stat": result.tvalues[param],
                            "p_value": result.pvalues[param],
                            "significant": result.pvalues[param] < 0.05,
                            "n_obs": int(result.nobs),
                        }
                    )
            except Exception as exc:
                print(f"Skipped {greek}/{bucket}: {exc}")

    return pd.DataFrame(records)


def interaction_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the full results table to a γ-coefficient matrix:
    rows = moneyness buckets, columns = interaction terms (regimes).

    Use this to produce the 4×3 heatmap in the paper.
    """
    interact = results_df[results_df["parameter"].str.startswith("interact_")].copy()
    interact["regime"] = interact["parameter"].str.replace("interact_", "", regex=False)
    return interact.pivot_table(
        index="moneyness", columns=["greek", "regime"], values="coef"
    ).round(6)
