"""Small EDA/plotting helpers for the shark attack dataset, used to validate
the "some activities carry more risk than others" hypothesis.
"""

import matplotlib.pyplot as plt
import pandas as pd


def top_activities(df, n=10):
    """Return the n most frequent activities (excluding 'Unknown').

    Used to identify which water activities have the highest attack volume.
    """
    counts = df.loc[df["activity"] != "Unknown", "activity"].value_counts()
    return counts.head(n)


def fatality_rate_by_activity(df, min_count=20, n=10):
    """Return fatality rate (%) for top activities, by frequency.

    `min_count` filters out activities with too few recorded attacks to
    give a statistically meaningful rate. Only returns activities with
    at least `min_count` documented attacks.

    Returns: Series of fatality rates (%) sorted descending.
    """
    known = df[(df["activity"] != "Unknown") & (df["fatal"] != "Unknown")]
    counts = known["activity"].value_counts()
    frequent = counts[counts >= min_count].index

    subset = known[known["activity"].isin(frequent)]
    rate = subset.groupby("activity")["fatal"].apply(lambda s: (s == "Y").mean() * 100)
    return rate.sort_values(ascending=False).head(n)


def plot_top_activities(df, n=10, ax=None):
    """Plot horizontal bar chart of top n activities by attack count.

    Args:
        df: cleaned shark attack dataframe
        n: number of top activities to show (default 10)
        ax: matplotlib axis (default: creates new figure)
    """
    counts = top_activities(df, n=n)
    ax = ax or plt.gca()
    counts.sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
    ax.set_xlabel("Number of attacks")
    ax.set_title(f"Top {n} activities by number of shark attacks")
    return ax


def plot_fatality_rate_by_activity(df, min_count=20, n=10, ax=None):
    """Plot horizontal bar chart of fatality rate by activity.

    Args:
        df: cleaned shark attack dataframe
        min_count: minimum attacks required to include activity (default 20)
        n: number of top activities to show (default 10)
        ax: matplotlib axis (default: creates new figure)
    """
    rate = fatality_rate_by_activity(df, min_count=min_count, n=n)
    ax = ax or plt.gca()
    rate.sort_values().plot(kind="barh", ax=ax, color="#d62728")
    ax.set_xlabel("Fatality rate (%)")
    ax.set_title(f"Fatality rate by activity (min. {min_count} recorded attacks)")
    return ax


def attacks_by_activity_and_country(df, activities, countries, n=8):
    """Crosstab of attack counts: activities (columns) by countries (rows).

    Shows the top `n` countries by total attack volume. Useful for
    identifying geographic + activity risk combinations.

    Args:
        df: cleaned shark attack dataframe
        activities: list of activity names to include
        countries: ignored; always uses top n countries by volume
        n: number of top countries to show (default 8)
    """
    top_countries = df["country"].value_counts().head(n).index
    subset = df[df["country"].isin(top_countries) & df["activity"].isin(activities)]
    return pd.crosstab(subset["country"], subset["activity"])
