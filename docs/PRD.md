# PRD: SPX Options Greek Sensitivity Across Market Regimes

## Current State

All modules, tests, and notebook stubs are built. The repo is ready to implement. Run notebooks in order: **02 → 01 → 03**.

```
src/features.py           ✅ built
src/regime.py             ✅ built
src/rolling_regression.py ✅ built  (5 tests passing)
src/panel.py              ✅ built
src/sensitivity.py        ✅ built  (4 tests passing)
src/plots.py              ✅ built
notebooks/01_*.ipynb      🔲 stub — needs implementation
notebooks/02_*.ipynb      🔲 stub — needs implementation
notebooks/03_*.ipynb      🔲 stub — needs implementation
data/regime_labels.parquet 🔲 created at runtime by notebook 02
```

---

## Problem Statement

Options trading strategies are fundamentally defined by their Greek exposure — delta, gamma, vega, and theta. Yet existing academic literature treats Greek sensitivities as stable properties of contracts, ignoring how the *relationship* between Greeks and market moves changes dramatically across market conditions. A high-gamma book in a calm market requires routine delta hedging. The same book during a volatility spike requires fundamentally different, more aggressive hedging behavior. There is no empirical characterization of how Greek sensitivities shift across formally identified market regimes in the SPX options market.

## Solution

A three-part empirical paper using SPX options data (1996–2019) that:

1. **Demonstrates structural instability** in the Greek-to-market relationship via rolling regressions, motivating a formal regime model
2. **Classifies market regimes** using a 4-state Gaussian Hidden Markov Model trained on options-market features
3. **Quantifies Greek sensitivity shifts** across regimes using a contract-level panel regression, stratified by moneyness and instrument type (puts vs. calls)

The result is a complete empirical characterization of how hedging requirements for SPX options change across market regimes — calm, rising stress, crisis, and recovery.

---

## User Stories

### Part 1 — Structural Instability

1. As a researcher, I want to run rolling OLS regressions of daily Greek changes on market returns and VIX changes, so that I can show the regression coefficients are non-constant over time.
2. As a researcher, I want to plot rolling regression coefficients as time series with confidence bands, so that I can visually demonstrate that Greek sensitivity spikes during known stress periods.
3. As a researcher, I want to test for structural breaks using a Chow test, so that the non-stationarity claim is empirically grounded rather than visual.
4. As a researcher, I want to show that a single unconditional regression produces misleading average coefficients, so that the motivation for a regime-switching model is clear to readers.
5. As a researcher, I want to produce a summary table of rolling coefficient statistics (mean, std, min, max) across known crisis sub-periods, so that I can quantify how much the relationship varies.

### Part 2 — Regime Classification

6. As a researcher, I want to engineer four daily features (VIX level, VRP, 21-day momentum, options bid-ask spread) from raw parquet files, so that the HMM has a principled, theoretically motivated input set.
7. As a researcher, I want to compute VRP as VIX minus 20-day realized volatility (both annualized), so that I capture the variance risk premium on a daily basis.
8. As a researcher, I want to compute 21-day cumulative SPX return from Fama-French factor data, so that I have a momentum signal without requiring a separate price series.
9. As a researcher, I want to compute the daily median options bid-ask spread from the cleaned SPX options data, so that I have a liquidity signal specific to the options market.
10. As a researcher, I want to train a 4-state Gaussian HMM on the engineered feature set, so that market regimes are identified data-driven rather than by arbitrary thresholds.
11. As a researcher, I want to label each of the 4 HMM states post-hoc by inspecting their feature distributions, so that states can be named interpretably (calm, rising stress, crisis, recovery).
12. As a researcher, I want to validate HMM state assignments against known crisis dates (Asian crisis 1997–98, dot-com 2000–02, GFC 2007–09, Euro debt 2011, China shock 2015–16, Q4 2018), so that the model's output is interpretable and credible.
13. As a researcher, I want to produce a timeline visualization of daily regime labels spanning 1996–2019, so that readers can see how the model classifies historical periods.
14. As a researcher, I want to run the HMM with 3 states as a robustness check, so that the choice of 4 states can be defended empirically.
15. As a researcher, I want to report the HMM transition probability matrix, so that I can describe regime persistence and switching dynamics.
16. As a researcher, I want to report feature distribution statistics (mean, std) by regime, so that readers can understand what characterizes each state.

### Part 3 — Greek Sensitivity Analysis

17. As a researcher, I want to build a contract-level panel from the raw SPX options data, so that I have enough observations to estimate regime-interacted regression coefficients reliably.
18. As a researcher, I want to filter contracts to the Primary Sample (20–45 DTE) and non-zero Greeks, so that the panel reflects actively hedged, liquid contracts.
19. As a researcher, I want to compute day-over-day ΔGreek (delta, gamma, vega, theta) within each contract, so that the dependent variable captures within-contract Greek dynamics.
20. As a researcher, I want to assign each panel observation its HMM regime label from Part 2, so that the panel and regime classification are linked.
21. As a researcher, I want to classify panel observations into moneyness buckets (Deep OTM: δ < 0.20, OTM: 0.20–0.40, ATM: 0.40–0.60) using absolute delta, so that I can stratify the sensitivity analysis.
22. As a researcher, I want to analyze puts and calls in separate regressions, so that I can capture the asymmetric Greek dynamics of crash protection instruments versus directional instruments.
23. As a researcher, I want to run the regime-interacted regression ΔGreek = α + β₁·ΔMarket + β₂·ΔVIX + Σγₖ·(Regimeₖ × ΔMarket) + ε for each Greek, moneyness bucket, and instrument type, so that I obtain a complete table of regime-conditional sensitivities.
24. As a researcher, I want the γₖ coefficients to be interpreted as the additional Greek sensitivity during regime k relative to the calm baseline, so that the hedging cost story is quantified precisely.
25. As a researcher, I want to report HC3-robust standard errors on all regressions, so that heteroskedasticity does not inflate significance.
26. As a researcher, I want to produce a 4×3 heatmap of γ coefficients (regimes × moneyness buckets) for each Greek, so that the cross-sectional pattern is visually apparent.
27. As a researcher, I want to run the same panel regression on the FOMC Window sample (5–20 DTE), so that I can report whether short-dated liquidity events produce different Greek sensitivities.
28. As a researcher, I want to compare put vs. call γ coefficients within the same moneyness bucket, so that I can show the asymmetry in crash protection demand across regimes.
29. As a researcher, I want to include contract fixed effects in the panel regression as a sensitivity check, so that time-invariant contract characteristics are absorbed.
30. As a researcher, I want to produce a summary table of all significant γ coefficients across Greeks, moneyness, and regimes, so that practitioners can extract the key hedging implications.

---

## Implementation Decisions

### Module Architecture

**`src/features.py` — Feature Engineering**
Builds the daily HMM input DataFrame from three sources: `data/cboeall1986.parquet` (VIX), `data/ff3.parquet` (SPX returns via mktrf+rf), `data/spx_clean.parquet` (options bid-ask spread). Outputs columns: `date, vix, vrp, momentum, opt_spread`. The leading rolling-window warm-up rows are dropped — output is complete with no NaNs.

**`src/regime.py` — HMM Classifier**
Wraps `hmmlearn.GaussianHMM`. Standardizes features (StandardScaler) before fitting. Key methods: `fit(features)`, `predict(features)`, `predict_named(features)`, `label_states(features)`, `state_summary(features)`, `transition_matrix()`, `crisis_validation(dates, labels)`. States are named post-hoc by VIX mean rank (lowest VIX = calm, highest = crisis). Random state 42 for reproducibility.

**`src/rolling_regression.py` — Rolling OLS**
`rolling_ols(y, X, window=126)` returns a date-indexed DataFrame of coefficients and SEs. `chow_test(y, X, breakpoint_idx)` returns F-stat and p-value. `rolling_coefficient_summary(rolling_df, coef_col, crisis_periods)` produces the summary table for the paper. Default window: 126 trading days (6 months).

**`src/panel.py` — Contract-Level Panel Builder**
Reads `data/spx_raw.parquet`, filters to DTE window and non-zero Greeks, sorts within each contract by date, diffs Greeks to get ΔGreek, drops first obs per contract and observations with date gaps > 5 calendar days. Merges regime labels and market features. Returns `(puts_df, calls_df)`. Contract ID: `(secid, strike_price, exdate, cp_flag)`.

**`src/sensitivity.py` — Greek Sensitivity Regressions**
`run_sensitivity_regression(panel, greek, moneyness)` fits the regime-interacted OLS with HC3 SEs. `run_all(panel)` iterates over all greek × moneyness combinations and returns a tidy results DataFrame. `interaction_table(results_df)` pivots γ coefficients into a matrix for the heatmap.

**`src/plots.py` — Visualizations**
`plot_rolling_coefficients(rolling_df, coef_col, se_col)` — rolling coefficient time series with crisis shading. `plot_regime_timeline(dates, named_labels)` — color-coded day-by-day timeline. `plot_sensitivity_heatmap(results_df, greek, instrument)` — γ heatmap. `plot_put_call_comparison(puts_results, calls_results, greek, moneyness)` — side-by-side bar chart.

### Key Technical Decisions

- **HMM features standardized before fitting** — VIX (~10–80) and momentum (~small decimals) have different scales; StandardScaler prevents scale-dominated covariance matrices.
- **SPX return derived from FF3** — `mktrf + rf` gives total SPX return without needing a separate price file.
- **Strike prices in spx_raw are ×1000** — e.g., 500000 = SPX at 500. Do not divide unless constructing moneyness from strike/spot.
- **Regime interaction baseline = calm** — all γₖ measure deviation from the calm period sensitivity.
- **Date gap filter ≤ 5 calendar days** — covers weekends (2 days) and single holidays (3 days); gaps > 5 signal a missing trading day within a contract and are dropped.
- **Puts and calls split by `cp_flag`** — 'P' = put, 'C' = call. Never aggregate across them.

### Data Dependencies

```
notebooks/02 → data/regime_labels.parquet → notebooks/03
data/cboeall1986.parquet, data/ff3.parquet, data/spx_clean.parquet → src/features.py
data/spx_raw.parquet → src/panel.py
```

---

## Testing Decisions

Good tests verify *external behavior* — outputs given inputs — not implementation details.

**`test_rolling_regression.py`** — 5 tests, no data I/O. Synthetic data only. Tests: date indexing, output length, convergence to full-sample OLS, Chow detects real break, Chow doesn't detect when stable.

**`test_sensitivity.py`** — 4 tests, no data I/O. Synthetic panel. Tests: correct coefficient count (const + ΔMarket + ΔVIX + interactions), puts/calls produce different γ coefficients, all greek×moneyness combos covered, HC3 SEs present.

**`test_features.py`** — 5 tests, reads parquets. Tests: VRP goes negative in crises, momentum is cumulative not mean, no NaNs in output, date range respected, correct output columns.

**`test_regime.py`** — 5 tests, reads parquets. Tests: labels in valid range, reproducibility with fixed seed, high-VIX days share a dominant state, label_states returns 4 names, transition matrix rows sum to 1.

**`test_panel.py`** — 5 tests, reads parquets. Tests: no NaN ΔGreek in output, no date gaps > MAX, DTE bounds respected, puts/calls disjoint, min_contract_obs enforced.

No tests for `plots.py` — visual correctness verified by inspection.

---

## Out of Scope

- **Live or forward-looking data**: All analysis is historical (1996–2019).
- **Trading strategy or backtesting**: The project characterizes Greek sensitivities; it does not construct or evaluate portfolios for trading.
- **Delta-hedged return calculation**: Considered and rejected in favor of direct Greek sensitivity analysis.
- **MSCI, equitytrove, all.parquet**: Not used. Equity-side analysis is out of scope.
- **Multi-index or individual equity options**: SPX index options only.
- **Causal inference**: All regressions are associational.
- **Publication formatting**: Tables and figures are research outputs, not publication-ready.

---

## Further Notes

- **Run order**: Notebook 02 must run before 03 — it produces `data/regime_labels.parquet`. Notebook 01 is independent.
- **Prior notebook** (`notebooks/00_reference_prior_regression.ipynb`): The original econometrics class project. Regressed monthly avg SPX IV on lagged FF5 factors + crisis dummy. Key findings: IV is AR(1) (β=0.82), market and momentum matter, crisis amplifies the market→IV relationship ~10x. Superseded by Part 1 of this project but kept for reference.
- **Counterintuitive VRP finding**: VRP is empirically lower during crisis (mean 2.81) than calm (mean 4.16). Realized vol spikes faster than implied vol can reprice. Worth highlighting in Part 2.
- **FOMC window rationale**: Short-dated options (5–20 DTE) become anomalously liquid around scheduled Fed meetings. Keeping them as a separate robustness test avoids contaminating primary results while still testing whether the event-driven liquidity pattern changes Greek sensitivity estimates.
- **Feature selection validation**: ANOVA Cohen d against labeled crisis periods — VIX (1.30), realized vol (1.19), opt_spread (0.65), momentum (0.50), VRP (0.26). VIX daily change (0.02) explicitly rejected.
- **Practitioner motivation**: QIS Derivatives trader at Morgan Stanley — "all options strategies are Greek exposure bets." This framing belongs in the paper's introduction.
