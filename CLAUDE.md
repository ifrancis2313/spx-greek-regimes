# CLAUDE.md — Project Brief

## What this project is

An academic research paper studying how the sensitivity of SPX option Greeks (delta, gamma, vega, theta) to market moves changes across market regimes. The practitioner motivation: a QIS Derivatives trader at Morgan Stanley noted that all options strategies are fundamentally Greek exposure bets — this paper quantifies how those bets behave differently depending on market conditions.

The research question: **Does the regime matter for hedging? How much more sensitive is gamma to a 1% market move during a crisis vs. a calm period?**

## Project structure

```
.
├── CLAUDE.md                              ← you are here
├── CONTEXT.md                             ← domain glossary (read this)
├── docs/
│   ├── PRD.md                             ← full specification (read this)
│   └── adr/
│       ├── 0001-hmm-regime-classification.md
│       └── 0002-contract-level-panel.md
├── data/                                  ← gitignored, parquet files live here
│   ├── cboeall1986.parquet                ← CBOE VIX + vol indices (daily, 1986–2026)
│   ├── spx_raw.parquet                    ← Raw SPX options (5.4M rows, 1996–2019)
│   ├── spx_clean.parquet                  ← Cleaned SPX options (90K rows, filtered)
│   ├── ff3.parquet                        ← Fama-French 3-factor + momentum (daily)
│   ├── ff5.parquet                        ← Fama-French 5-factor + momentum (daily)
│   ├── msci.parquet                       ← MSCI index data (not used in main analysis)
│   ├── fullpanelmsci.parquet              ← MSCI full panel (not used)
│   ├── mscidictionary.parquet             ← MSCI dictionary (not used)
│   ├── equitytrove.parquet                ← Equity data (not used in main analysis)
│   ├── Fama_French_Data.parquet           ← Duplicate FF data (not used)
│   ├── all.parquet                        ← Large combined dataset (not used)
│   └── regime_labels.parquet             ← OUTPUT of notebook 02 (created at runtime)
├── src/                                   ← all importable modules
│   ├── features.py                        ← daily HMM feature engineering
│   ├── regime.py                          ← 4-state Gaussian HMM classifier
│   ├── rolling_regression.py              ← rolling OLS + Chow structural break test
│   ├── panel.py                           ← contract-level panel builder
│   ├── sensitivity.py                     ← regime-interacted Greek regressions
│   └── plots.py                           ← shared visualizations
├── notebooks/
│   ├── archive/
│   │   └── 00_reference_prior_regression.ipynb ← prior coursework (reference only, not part of the paper)
│   ├── 01_structural_instability.ipynb    ← Part 1: rolling regressions
│   ├── 02_regime_classification.ipynb     ← Part 2: HMM (run this first)
│   └── 03_greek_sensitivity.ipynb         ← Part 3: panel regressions
├── tests/
│   ├── test_features.py
│   ├── test_regime.py
│   ├── test_panel.py
│   ├── test_rolling_regression.py         ← 5 passing, no data I/O
│   └── test_sensitivity.py               ← 4 passing, no data I/O
├── requirements.txt
└── .gitignore
```

## Three-part paper structure

### Part 1 — Structural Instability (`notebooks/01_structural_instability.ipynb`)
Rolling OLS regressions showing that Greek sensitivity to market moves is non-stationary — coefficients spike during crisis and compress during calm. This motivates the HMM in Part 2. Uses Chow test for formal structural break detection.

### Part 2 — Regime Classification (`notebooks/02_regime_classification.ipynb`)
4-state Gaussian HMM trained on four daily features:
- `vix` — CBOE VIX level
- `vrp` — VIX minus 20-day realized vol (variance risk premium)
- `momentum` — 21-day cumulative SPX return
- `opt_spread` — median options bid-ask spread

**Run this notebook first.** It produces `data/regime_labels.parquet` which Part 3 depends on.

Anticipated states: calm / rising_stress / crisis / recovery (labeled post-hoc by VIX mean rank).

### Part 3 — Greek Sensitivity (`notebooks/03_greek_sensitivity.ipynb`)
Contract-level panel from `spx_raw.parquet`. For each consecutive-day observation of a contract, computes ΔGreek and runs:

```
ΔGreek = α + β1·ΔMarket + β2·ΔVIX + Σγk·(Regimek × ΔMarket) + ε
```

γk coefficients = additional sensitivity vs. calm baseline.

- Primary sample: 20–45 DTE (active hedging window where market makers operate)
- Puts and calls analyzed separately
- Moneyness buckets: deep_otm (δ<0.20), otm (0.20–0.40), atm (0.40–0.60)
- Robustness: 5–20 DTE (FOMC window), tested separately

## Key data facts

| File | Rows | Dates | Notes |
|---|---|---|---|
| `spx_raw.parquet` | 5,406,736 | 1996–2019 | 0.6GB in memory. 96.4% non-zero Greeks. |
| `spx_clean.parquet` | 90,554 | 1996–2019 | Pre-filtered version from prior regression project |
| `cboeall1986.parquet` | 10,656 | 1986–2026 | VIX starts 1993; VXO available from 1986 |
| `ff3.parquet` | 26,171 | 1926–2026 | Daily. Includes `umd` (momentum). Use `mktrf + rf` for SPX return. |

Strike prices in `spx_raw.parquet` are in units of 1/1000 (e.g., 500000 = SPX at 500). Delta and other Greeks are pre-computed by OptionMetrics.

## How to run

```bash
# Activate venv
source .venv/bin/activate

# Run tests (fast — no data I/O required)
python -m pytest tests/test_rolling_regression.py tests/test_sensitivity.py -v

# Run data-dependent tests (slow — reads parquets)
python -m pytest tests/ -v

# Open notebooks
jupyter notebook notebooks/
```

Run notebooks in order: 02 → then 01 and 03 can run independently.

## Key design decisions (see docs/adr/ for full rationale)

- **HMM over VIX thresholds**: Data-driven, reproducible, defensible in peer review
- **4 states over 3**: Captures calm/rising stress/crisis/recovery arc — recovery differs from both calm and crisis
- **Contract-level panel over monthly aggregation**: 5M+ observations vs 288 monthly — much more statistical power
- **20–45 DTE primary sample**: Active hedging window; excludes near-expiry noise and LEAPS
- **HC3 robust SEs throughout**: Expected heteroskedasticity from volatility clustering
- **Puts and calls separate**: A 0.20-delta put is not the same instrument as a 0.20-delta call

## What is NOT in scope

- Live/forward-looking data — historical 1996–2019 only
- Trading strategy or backtest
- Delta-hedged return calculation (an earlier idea, rejected in favor of Greek sensitivities)
- MSCI, equitytrove, or equity-side analysis
- Publication formatting

## Python environment

Virtual environment at `.venv/`. Python 3.13. Key packages: pandas, numpy, statsmodels, hmmlearn, scikit-learn, scipy, matplotlib, seaborn, pyarrow. Install with `pip install -r requirements.txt`.
