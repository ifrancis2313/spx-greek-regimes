# SPX Greek Sensitivity Across Market Regimes

**Repo:** https://github.com/ifrancis2313/spx-greek-regimes

Does market regime change how sensitive SPX option Greeks are to market moves — and does that matter for hedging? A QIS derivatives trader's observation that "every options strategy is fundamentally a Greek exposure bet" is the practitioner anchor for this project: it quantifies how those bets behave differently in calm markets versus crises.

**Research question:** How much more sensitive is gamma to a 1% market move during a crisis than during a calm period?

## Key findings

Using a 4-state Hidden Markov Model to classify 6,022 trading days (1996–2019) into regimes, and a contract-level panel of ~1.2M (contract, day) observations:

- **ATM put gamma is ~5x more sensitive to market moves in crisis than in calm regimes** (calm-baseline coefficient -0.00071 vs. an additional -0.00288 crisis interaction — a combined crisis sensitivity of roughly -0.00359).
- **ATM call gamma sensitivity nearly reverses sign in crisis** (calm baseline +0.0123, crisis interaction -0.0127, leaving the crisis-period sensitivity close to zero/slightly negative) — direct evidence that puts and calls are not symmetric instruments under stress, consistent with treating them separately throughout this project.
- **The sensitivity relationship is non-stationary**: a Chow test for a structural break at the start of the GFC (2007-07-01) strongly rejects coefficient stability (F = 29.68, p ≈ 0), motivating the regime-based approach over a single unconditional regression.
- Regime breakdown across the sample: 1,854 days rising-stress, 1,762 recovery, 1,495 calm, 911 crisis.

Full coefficient tables (all 4 Greeks × 3 moneyness buckets × 4 regimes, puts and calls separately, plus a 5–20 DTE FOMC-window robustness check) are in the executed output of `notebooks/03_greek_sensitivity.ipynb`.

## Project structure

Three-part paper, each part a notebook backed by tested `src/` modules:

| Part | Notebook | What it does |
|---|---|---|
| 1 — Structural Instability | `01_structural_instability.ipynb` | Rolling OLS shows Greek sensitivity to market moves is non-stationary; Chow test confirms a structural break at the GFC. Motivates Part 2. |
| 2 — Regime Classification | `02_regime_classification.ipynb` | 4-state Gaussian HMM on VIX level, variance risk premium, 21-day momentum, and options bid-ask spread. States labeled post-hoc (calm / rising stress / crisis / recovery) and validated against known crisis dates. Outputs `data/regime_labels.parquet`, which Part 3 depends on. |
| 3 — Greek Sensitivity Panel | `03_greek_sensitivity.ipynb` | Contract-level panel (20–45 DTE primary sample; puts/calls separate; moneyness buckets deep OTM / OTM / ATM). Regresses ΔGreek on ΔMarket, ΔVIX, and regime interactions to get the γ coefficients above. 5–20 DTE FOMC window tested as a robustness check. |

See [`CONTEXT.md`](CONTEXT.md) for the domain glossary, [`docs/PRD.md`](docs/PRD.md) for the full spec, and [`docs/adr/`](docs/adr) for the rationale behind the HMM and panel design choices.

`notebooks/archive/00_reference_prior_regression.ipynb` is earlier coursework that motivated this project (it independently found the same "crisis amplifies market sensitivity" pattern, on a much smaller monthly dataset) — kept for reference, not part of the three-part paper above.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fast tests (no data I/O)
python -m pytest tests/test_rolling_regression.py tests/test_sensitivity.py -v

# Full test suite (reads parquet files — needs data/ populated locally)
python -m pytest tests/ -v

# Notebooks must run in this order — 02 produces the regime labels 01 and 03 depend on:
jupyter nbconvert --to notebook --execute --inplace notebooks/02_regime_classification.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_structural_instability.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_greek_sensitivity.ipynb
```

## Data

Source data (`data/`, gitignored — not included in this repo) is OptionMetrics SPX options (1996–2019, 5.4M raw rows), CBOE VIX/vol indices, and Fama-French factors. The options data is licensed and cannot be redistributed; only code and notebook outputs are public here.

## Scope

Historical 1996–2019 data only — no live/forward-looking analysis, no trading strategy or backtest, and no delta-hedged return calculation (considered and rejected in favor of Greek sensitivities directly — see ADRs).
