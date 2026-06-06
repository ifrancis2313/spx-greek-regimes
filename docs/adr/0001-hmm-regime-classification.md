# ADR 0001: 4-State Gaussian HMM for Regime Classification

## Status
Accepted

## Context
The project requires classifying every trading day (1996–2019, ~6,040 days) into a market regime. The regime label is then used as an interaction term in the Part 3 Greek sensitivity regressions. Two approaches were considered: rule-based VIX thresholds and a data-driven Hidden Markov Model.

Rule-based thresholds (e.g., VIX < 15 = calm, VIX > 35 = crisis) are transparent but arbitrary. The cutoffs are researcher-chosen and not derived from the data, which creates a degrees-of-freedom problem and is difficult to defend in peer review. They also ignore all information beyond VIX level.

## Decision
Use a 4-state Gaussian HMM trained on four daily features: VIX level, Variance Risk Premium, 21-day momentum, and options bid-ask spread. Features were selected by ANOVA discrimination against labeled crisis periods (Cohen d ranging from 0.26 to 1.30) and validated for theoretical connection to prior regression findings.

Four states were chosen over three to capture the full crisis arc: calm → rising stress → crisis → recovery. The recovery state is empirically meaningful — Greek sensitivity dynamics during the unwinding of a crisis differ from both crisis peak and calm, and collapsing them loses that distinction.

## Consequences
- Regime labels are data-driven and reproducible, not researcher-chosen
- Four states require estimating more parameters than three; with 6,040 daily observations this is well-identified
- States must be labeled post-hoc by inspecting their feature distributions and alignment with known crisis dates (dot-com, GFC, Euro debt, etc.)
- Short-dated (5–20 DTE) robustness tests use the same regime labels, so HMM output is a shared dependency across all downstream analyses
- The number of states (4) is a modeling assumption that should be reported with robustness checks using 3 states
