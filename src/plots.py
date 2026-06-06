"""
Shared visualization module. All figures return plt.Figure objects
so callers decide whether to show or save them.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

REGIME_COLORS = {
    "calm": "#4CAF50",
    "rising_stress": "#FF9800",
    "crisis": "#F44336",
    "recovery": "#2196F3",
}

KNOWN_CRISES = [
    ("1997-07-01", "1998-10-31", "Asian+LTCM"),
    ("2000-03-01", "2002-10-31", "Dot-com"),
    ("2007-07-01", "2009-06-30", "GFC"),
    ("2011-07-01", "2011-12-31", "Euro debt"),
    ("2015-08-01", "2016-02-29", "China shock"),
    ("2018-10-01", "2018-12-31", "Q4 2018"),
]


def plot_rolling_coefficients(
    rolling_df: pd.DataFrame,
    coef_col: str,
    se_col: str,
    title: str = "Rolling OLS Coefficient",
    annotate_crises: bool = True,
) -> plt.Figure:
    """
    Time series of a rolling OLS coefficient with ±2 SE confidence band.
    Known crisis windows are shaded in light red if annotate_crises=True.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    coefs = rolling_df[coef_col]
    ses = rolling_df[se_col]

    ax.plot(rolling_df.index, coefs, color="steelblue", linewidth=1.5, label="Coefficient")
    ax.fill_between(
        rolling_df.index, coefs - 2 * ses, coefs + 2 * ses, alpha=0.2, color="steelblue"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    if annotate_crises:
        for start, end, label in KNOWN_CRISES:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color="firebrick")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Coefficient")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_regime_timeline(
    dates: pd.Series,
    named_labels: pd.Series,
) -> plt.Figure:
    """
    Horizontal timeline colored by regime label.
    Each trading day is rendered as a thin vertical bar.
    """
    fig, ax = plt.subplots(figsize=(16, 2.5))

    for date, regime in zip(dates, named_labels):
        color = REGIME_COLORS.get(regime, "gray")
        ax.axvline(date, color=color, linewidth=0.6, alpha=0.8)

    patches = [
        mpatches.Patch(color=c, label=r.replace("_", " ").title())
        for r, c in REGIME_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    ax.set_title("Market Regime Classification — SPX Options 1996–2019", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_sensitivity_heatmap(
    results_df: pd.DataFrame,
    greek: str,
    instrument: str = "puts",
) -> plt.Figure:
    """
    Heatmap of γ (interaction) coefficients for one Greek across
    moneyness buckets (rows) and regimes (columns).
    Red = positive (amplified sensitivity), Blue = negative.
    """
    interact = results_df[
        results_df["parameter"].str.startswith("interact_")
        & (results_df["greek"] == greek)
    ].copy()
    interact["regime"] = interact["parameter"].str.replace("interact_", "", regex=False)

    pivot = interact.pivot(index="moneyness", columns="regime", values="coef")

    # Order moneyness rows consistently
    row_order = [r for r in ["deep_otm", "otm", "atm"] if r in pivot.index]
    pivot = pivot.reindex(row_order)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".5f",
        center=0,
        cmap="RdBu_r",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        f"Regime-Conditional {greek.capitalize()} Sensitivity — {instrument.capitalize()}",
        fontweight="bold",
    )
    ax.set_xlabel("Regime (vs. calm baseline)")
    ax.set_ylabel("Moneyness")
    plt.tight_layout()
    return fig


def plot_put_call_comparison(
    puts_results: pd.DataFrame,
    calls_results: pd.DataFrame,
    greek: str,
    moneyness: str,
) -> plt.Figure:
    """
    Side-by-side bar chart comparing γ coefficients for puts vs. calls
    at the same Greek × moneyness cell.
    """
    def _extract(df):
        return (
            df[
                df["parameter"].str.startswith("interact_")
                & (df["greek"] == greek)
                & (df["moneyness"] == moneyness)
            ]
            .assign(regime=lambda d: d["parameter"].str.replace("interact_", "", regex=False))
            .set_index("regime")[["coef", "se"]]
        )

    p = _extract(puts_results)
    c = _extract(calls_results)
    regimes = sorted(set(p.index) | set(c.index))
    x = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [p.loc[r, "coef"] if r in p.index else 0 for r in regimes],
           width, yerr=[p.loc[r, "se"] if r in p.index else 0 for r in regimes],
           label="Puts", color="firebrick", alpha=0.75, capsize=4)
    ax.bar(x + width / 2, [c.loc[r, "coef"] if r in c.index else 0 for r in regimes],
           width, yerr=[c.loc[r, "se"] if r in c.index else 0 for r in regimes],
           label="Calls", color="steelblue", alpha=0.75, capsize=4)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").title() for r in regimes])
    ax.set_title(
        f"Put vs. Call γ Coefficients — {greek.capitalize()} / {moneyness.replace('_', ' ').title()}",
        fontweight="bold",
    )
    ax.set_ylabel("γ Coefficient (vs. calm baseline)")
    ax.legend()
    plt.tight_layout()
    return fig
