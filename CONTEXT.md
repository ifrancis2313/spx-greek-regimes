# Context: SPX Options Greek Sensitivity Across Market Regimes

## Domain Glossary

**Greek Sensitivity**
The rate of change of an option Greek (delta, gamma, vega, theta) in response to a unit change in an underlying market variable (SPX daily return or VIX), estimated within a specific market regime. The central quantity of interest in Part 3 of this project.

**Market Regime**
A latent state of market conditions identified by the Hidden Markov Model. Characterized by a distinct joint distribution of VIX level, Variance Risk Premium, market momentum, and options bid-ask spread. The project uses 4 regimes, anticipated to correspond to: calm, rising stress, crisis, and recovery.

**Variance Risk Premium (VRP)**
The spread between implied volatility (VIX, annualized %) and 20-day realized volatility (annualized %), computed daily. Represents the premium option sellers earn for bearing volatility risk. Empirically lower during crisis periods than calm — because realized volatility spikes faster than implied volatility can adjust.

**Volatility Surface Skew**
The spread between the average implied volatility of OTM puts (abs_delta 0.05–0.35) and ATM options (abs_delta 0.35–0.65) at comparable maturities. Measures the excess demand for crash protection embedded in the options market.

**Structural Instability**
The property that regression coefficients relating Greek changes to market moves are non-constant over time. Documented in Part 1 via rolling regressions. The empirical motivation for regime-based modeling — if the relationship were stable, a single unconditional regression would suffice.

**Moneyness Bucket**
A categorical grouping of option contracts by absolute delta: Deep OTM (abs_delta < 0.20), OTM (0.20–0.40), ATM (0.40–0.60). Puts and calls are analyzed separately within each bucket — a 0.20-delta put and a 0.20-delta call are not treated as equivalent instruments.

**Contract-Level Panel**
The dataset formed by tracking individual option contracts — identified by (secid, strike_price, exdate, cp_flag) — across consecutive trading days. The unit of observation is a (contract, day) pair. Day-over-day changes in Greek values (ΔGreek) are computed within each contract and serve as the dependent variable in Part 3.

**Primary Sample**
Option contracts with 20–45 days to expiration. This is the "active hedging window" where market maker activity concentrates and Greek dynamics are most practically relevant to the hedging story. Contracts outside this window are excluded from main regressions.

**FOMC Window**
Option contracts with 5–20 days to expiration, which become anomalously liquid around scheduled Federal Open Market Committee meetings due to event-driven demand. Analyzed separately as a robustness check rather than included in the Primary Sample, to avoid contaminating the baseline Greek sensitivity estimates.

**Rolling Regression**
An OLS regression estimated over a fixed-width sliding window of daily observations (e.g., 126 trading days / 6 months). Used in Part 1 to plot time-varying regression coefficients and demonstrate that Greek sensitivity to market moves is non-stationary — spiking during stress periods and collapsing during calm.

**HMM Features**
The four daily input variables fed to the Hidden Markov Model: VIX level, Variance Risk Premium, 21-day cumulative SPX return (momentum), and median options bid-ask spread. Selected based on empirical ANOVA discrimination against labeled crisis periods and theoretical connection to the Part 1 regression findings.
