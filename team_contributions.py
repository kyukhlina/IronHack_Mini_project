"""Functions from teammates' own explorations, reused in the group notebook
(shark_attacks_analysis_group.ipynb).

Kept separate from `cleaning.py`/`eda.py` (Aroa's individual pipeline, fully
documented there) so it's clear which code is whose. Functions below are
attributed by section; only Aroa's own functions carry full docstrings —
teammates' functions are left close to their original form, with a short
comment crediting the source.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Berta's contribution: her basic text-cleaning helper (upper + strip),
# used in her sex-based attack-count analysis (see group notebook addendum).
# ---------------------------------------------------------------------------

def clear_data(series):
    # Berta's cleaning function, as written in her notebook.
    return series.str.upper().str.strip()


# ---------------------------------------------------------------------------
# Kseniia's contribution: choosing an analysis window based on completeness,
# and descriptive statistics per country/year.
# ---------------------------------------------------------------------------

def completeness_by_year_cutoff(df, year_col, cutoffs, columns=None):
    # Kseniia's technique: compare % non-null across candidate year cutoffs
    # instead of picking a time window arbitrarily.
    columns = columns or df.columns.tolist()
    rows = {}
    for cutoff in cutoffs:
        subset = df[df[year_col] > cutoff]
        pct_complete = (subset[columns].notna().mean() * 100).round(1)
        pct_complete["n_rows"] = len(subset)
        rows[f"> {cutoff}"] = pct_complete
    return pd.DataFrame(rows).T


def country_year_descriptive_stats(df, country_col, year_col, countries):
    # Kseniia's technique: run .describe() on the year x country pivot to
    # surface volatility (std), not just a raw trend line.
    subset = df[df[country_col].isin(countries)]
    pivot = pd.pivot_table(
        subset, index=year_col, columns=country_col, aggfunc="size", fill_value=0
    )
    return pivot, pivot.describe().round(2)
