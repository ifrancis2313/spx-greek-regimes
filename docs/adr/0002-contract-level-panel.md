# ADR 0002: Contract-Level Panel Over Monthly Aggregation

## Status
Accepted

## Context
Greek sensitivity analysis requires a dependent variable (ΔGreek) and enough observations to estimate regime-interacted regression coefficients. Two aggregation levels were considered: monthly averages (288 months × moneyness buckets) and a contract-level panel (individual option contracts tracked day-over-day).

Monthly aggregation was rejected because the cleaned dataset (spx_clean.parquet) averages only 1–2 contracts per day in ATM and ITM buckets, making daily-to-monthly aggregation too noisy and sparse for reliable inference.

## Decision
Use the raw options dataset (spx_raw.parquet, 5.4M rows, 230,473 unique contracts) as a contract-level panel. The unit of observation is a (contract, day) pair. Day-over-day ΔGreek is computed by sorting within each contract by date and differencing. Contracts with fewer than 2 consecutive observations are dropped.

The contract identifier is (secid, strike_price, exdate, cp_flag). The panel includes all contracts in the Primary Sample (20–45 DTE) with non-zero Greeks (96.4% of raw observations).

## Consequences
- ~5M panel observations versus ~288 monthly observations — substantially more statistical power
- Contract fixed effects can be included to absorb time-invariant contract characteristics
- Day-over-day ΔGreek is mechanically smooth within a contract (Black-Scholes is continuous), reducing noise relative to cross-sectional aggregation
- The raw dataset (0.6GB in memory) is manageable but requires careful memory management during panel construction
- Puts and calls are analyzed in separate regressions — cp_flag is a split variable, not a covariate
