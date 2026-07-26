"""
Reusable, testable plotting functions shared across the EDA notebook and
later evaluation phases. Every function follows the same contract: given
a DataFrame (and sometimes a save path), produce a matplotlib Figure,
optionally persist it to disk, and return the Figure so callers in a
notebook can also just display it inline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.rcParams["figure.dpi"] = 100


class PlottingError(Exception):
    """Raised when a plotting function is given data it cannot render."""


def _ensure_parent_dir(save_path: Path) -> None:
    """Create the parent directory for a figure output path if missing."""
    save_path.parent.mkdir(parents=True, exist_ok=True)


def _require_columns(df: pd.DataFrame, columns: List[str], fn_name: str) -> None:
    """Raise PlottingError with a clear message if required columns are absent."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise PlottingError(f"{fn_name} requires column(s) {missing}, not found in DataFrame.")


def _save_and_return(fig: plt.Figure, save_path: Optional[Path]) -> plt.Figure:
    """Shared save-and-return tail used by every plotting function below."""
    if save_path is not None:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Saved figure to %s", save_path)
    return fig


def plot_trip_duration_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot the distribution of trip_duration.

    Business insight: trip duration is almost always heavily right-skewed
    (most trips are short, a long tail of longer trips, plus a handful of
    data-entry errors near 0 seconds or many hours). This shape is why
    Phase 8 evaluates duration models on log-transformed targets or RMSLE
    rather than raw RMSE alone — a few extreme outliers would otherwise
    dominate the loss.

    Args:
        df: DataFrame containing a 'trip_duration' column, in seconds.
        save_path: If provided, the figure is saved here as a PNG.
        log_scale: If True, also plots log1p(trip_duration) side by side,
            which is usually far closer to a normal distribution.

    Returns:
        The matplotlib Figure.
    """
    _require_columns(df, ["trip_duration"], "plot_trip_duration_distribution")

    ncols = 2 if log_scale else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    axes = np.atleast_1d(axes)

    axes[0].hist(df["trip_duration"], bins=100, color="#377EB8")
    axes[0].set_title("Trip duration (raw seconds)")
    axes[0].set_xlabel("trip_duration (s)")
    axes[0].set_ylabel("count")

    if log_scale:
        axes[1].hist(np.log1p(df["trip_duration"]), bins=100, color="#4DAF4A")
        axes[1].set_title("Trip duration (log1p seconds)")
        axes[1].set_xlabel("log1p(trip_duration)")
        axes[1].set_ylabel("count")

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_pickup_hotspots(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    sample_size: int = 50_000,
    random_state: int = 42,
) -> plt.Figure:
    """
    Scatter-plot a sample of pickup coordinates to reveal geographic
    demand hotspots.

    Business insight: dense clusters here (typically Manhattan, plus
    airport terminals) are exactly what Phase 6's K-Means zone clustering
    needs to capture well — this plot is the visual justification for
    choosing K-Means over a naive equal-area grid.

    Args:
        df: DataFrame with 'pickup_longitude' and 'pickup_latitude'.
        save_path: If provided, the figure is saved here as a PNG.
        sample_size: Plotting all 1.45M points is unreadable and slow;
            a random sample preserves the density pattern at a fraction
            of the cost.
        random_state: Seed for the sample, for reproducible figures.

    Returns:
        The matplotlib Figure.
    """
    _require_columns(df, ["pickup_longitude", "pickup_latitude"], "plot_pickup_hotspots")

    sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        sample["pickup_longitude"],
        sample["pickup_latitude"],
        s=1,
        alpha=0.15,
        color="#E41A1C",
    )
    ax.set_title(f"Pickup locations (sample of {len(sample):,} trips)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_trips_by_hour(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot trip counts by pickup hour of day.

    Business insight: this is what motivates the rush-hour and
    cyclic-time features built in Phase 5 — if pickup volume clearly
    peaks around 8am and 6pm, a model that only sees a linear 'hour'
    feature (where hour 23 and hour 0 look maximally different, despite
    being one hour apart) is throwing away information a cyclic
    encoding would preserve.

    Args:
        df: DataFrame with a 'pickup_datetime' column (datetime64 dtype).
        save_path: If provided, the figure is saved here as a PNG.

    Returns:
        The matplotlib Figure.
    """
    _require_columns(df, ["pickup_datetime"], "plot_trips_by_hour")

    counts = df["pickup_datetime"].dt.hour.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color="#984EA3")
    ax.set_title("Trip counts by pickup hour")
    ax.set_xlabel("hour of day (0-23)")
    ax.set_ylabel("trip count")
    ax.set_xticks(range(0, 24))

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_missing_value_summary(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot the percentage of missing values per column.

    Business insight: confirms whether Phase 4's cleaning step needs an
    imputation or row-dropping strategy at all. The raw NYC dataset is
    typically very clean on this dimension, but this check should never
    be skipped or assumed.

    Args:
        df: Any DataFrame to audit for missingness.
        save_path: If provided, the figure is saved here as a PNG.

    Returns:
        The matplotlib Figure.
    """
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(missing_pct))))
    ax.barh(missing_pct.index, missing_pct.values, color="#FF7F00")
    ax.set_title("Missing values by column (%)")
    ax.set_xlabel("% missing")
    ax.invert_yaxis()

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a correlation heatmap over numeric columns.

    Business insight: a quick sanity check before modeling — e.g.
    confirming that raw pickup/dropoff coordinates alone have weak
    linear correlation with trip_duration, which is the justification
    for engineering a proper haversine distance feature in Phase 5
    rather than feeding raw lat/lon into a linear baseline.

    Args:
        df: DataFrame to correlate. Non-numeric columns are dropped
            automatically.
        save_path: If provided, the figure is saved here as a PNG.

    Returns:
        The matplotlib Figure.

    Raises:
        PlottingError: If fewer than 2 numeric columns are present.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise PlottingError(
            "plot_correlation_heatmap requires at least 2 numeric columns, "
            f"found {numeric_df.shape[1]}."
        )

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(0.6 * len(corr.columns) + 2, 0.6 * len(corr.columns) + 2))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
    ax.set_title("Correlation heatmap (numeric columns)")

    fig.tight_layout()
    return _save_and_return(fig, save_path)





def plot_pickup_zones(
    df: pd.DataFrame,
    zone_column: str = "pickup_zone",
    save_path: Optional[Path] = None,
    sample_size: int = 50_000,
    random_state: int = 42,
) -> plt.Figure:
    """
    Scatter-plot a sample of pickups colored by assigned K-Means zone.

    Business insight: this is the visual sanity check for Phase 6 -
    zones should look like sensible, roughly contiguous geographic
    regions (dense, small zones over Manhattan; larger zones covering
    sparser outer-borough areas), not scattered or overlapping noise.
    If zones look visually incoherent, that's a signal to revisit
    n_clusters or the longitude correction before trusting Phase 7's
    demand aggregation built on top of them.

    Args:
        df: DataFrame with pickup coordinates and a zone column.
        zone_column: Column containing integer/categorical zone ids.
        save_path: If provided, the figure is saved here as a PNG.
        sample_size: Points to sample before plotting (see
            plot_pickup_hotspots for why sampling matters here).
        random_state: Seed for the sample, for reproducible figures.

    Returns:
        The matplotlib Figure.
    """
    _require_columns(df, ["pickup_longitude", "pickup_latitude", zone_column], "plot_pickup_zones")

    sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        sample["pickup_longitude"],
        sample["pickup_latitude"],
        c=sample[zone_column].astype(int),
        cmap="tab20",
        s=3,
        alpha=0.4,
    )
    ax.set_title(f"Pickup zones (K-Means, sample of {len(sample):,} trips)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    fig.colorbar(scatter, ax=ax, shrink=0.7, label="zone id")

    fig.tight_layout()
    return _save_and_return(fig, save_path)