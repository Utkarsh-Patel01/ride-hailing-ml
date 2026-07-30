"""
Clusters pickup coordinates into a fixed set of operational zones used
by the demand-forecasting pipeline (Phase 7 onward). See Phase 6's
write-up for the K-Means vs. DBSCAN reasoning behind this design.

PickupZoneClusterer bundles the fitted KMeans model together with the
longitude correction factor computed at fit time, so training and
inference can never disagree about how coordinates were preprocessed -
the same failure mode build_features.py guards against for temporal
and spatial features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

ZONE_COLUMN = "pickup_zone"


class PickupZoneClusterer:
    """
    Fits and applies K-Means clustering to pickup coordinates, with a
    built-in longitude correction for NYC's latitude-dependent distance
    distortion. Persistable as a single unit via save()/load(), so the
    correction factor can never drift out of sync with the fitted model.
    """

    def __init__(self, n_clusters: int, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.longitude_correction_factor: Optional[float] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "PickupZoneClusterer":
        """
        Fit K-Means on pickup coordinates.

        Args:
            df: DataFrame with 'pickup_latitude' and 'pickup_longitude'.

        Returns:
            self, so this can be chained: clusterer = PickupZoneClusterer(30).fit(df)
        """
        lat, lon = df["pickup_latitude"], df["pickup_longitude"]

        # Fix the correction factor at fit time from the training
        # data's mean latitude - it must stay constant afterward so
        # every future prediction uses an identical coordinate space.
        self.longitude_correction_factor = float(np.cos(np.radians(lat.mean())))

        coords = self._prepare_coords(lat, lon)
        self.model.fit(coords)
        self.is_fitted = True

        logger.info(
            "Fitted %d pickup zones on %s trips (inertia=%.1f)",
            self.n_clusters, f"{len(df):,}", self.model.inertia_,
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Assign each row's pickup coordinates to the nearest fitted zone.

        Args:
            df: DataFrame with 'pickup_latitude' and 'pickup_longitude'.
                Works identically for 1.45M training rows or a single
                inference-time row.

        Returns:
            Array of integer zone ids, one per row.

        Raises:
            RuntimeError: If called before fit() or load().
        """
        if not self.is_fitted:
            raise RuntimeError("PickupZoneClusterer must be fit() or load()ed before predict().")

        coords = self._prepare_coords(df["pickup_latitude"], df["pickup_longitude"])
        return self.model.predict(coords)

    def _prepare_coords(self, lat: pd.Series, lon: pd.Series) -> np.ndarray:
        """Apply the fixed longitude correction and stack into a coordinate array."""
        corrected_lon = lon * self.longitude_correction_factor
        return np.column_stack([lat.to_numpy(), corrected_lon.to_numpy()])

    def save(self, path: Path) -> None:
        """Persist this object (model + correction factor) as a single artifact."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved pickup zone clusterer to %s", path)

    @staticmethod
    def load(path: Path) -> "PickupZoneClusterer":
        """Load a previously saved PickupZoneClusterer."""
        clusterer = joblib.load(path)
        if not isinstance(clusterer, PickupZoneClusterer):
            raise TypeError(f"{path} did not contain a PickupZoneClusterer instance.")
        return clusterer


def assign_pickup_zone(df: pd.DataFrame, clusterer: PickupZoneClusterer) -> pd.DataFrame:
    """
    Add a pickup_zone column to df using a fitted clusterer.

    Args:
        df: DataFrame with pickup coordinate columns.
        clusterer: A fitted (or loaded) PickupZoneClusterer.

    Returns:
        A copy of df with an added 'pickup_zone' column (category dtype).
    """
    df = df.copy()
    zone_ids = clusterer.predict(df)
    df[ZONE_COLUMN] = pd.Categorical(zone_ids, categories=range(clusterer.n_clusters))
    return df


def evaluate_k_range(
    df: pd.DataFrame,
    k_values: List[int],
    sample_size: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Exploratory tool: compute inertia and silhouette score across a
    range of k values, on a sample (full-dataset silhouette scoring on
    1.45M rows is prohibitively slow - O(n^2) pairwise distances).

    Not called by the production pipeline; run manually from a notebook
    or the command line when reconsidering the n_clusters value in
    config.yaml.

    Args:
        df: DataFrame with pickup coordinate columns.
        k_values: Candidate cluster counts to evaluate.
        sample_size: Rows to sample before fitting, for speed.
        random_state: Seed for both sampling and clustering.

    Returns:
        DataFrame with one row per k: columns 'k', 'inertia', 'silhouette'.
    """
    sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    lat, lon = sample["pickup_latitude"], sample["pickup_longitude"]
    corrected_lon = lon * np.cos(np.radians(lat.mean()))
    coords = np.column_stack([lat.to_numpy(), corrected_lon.to_numpy()])

    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(coords)
        score = silhouette_score(coords, km.labels_, sample_size=10_000, random_state=random_state)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": score})
        logger.info("k=%d: inertia=%.1f, silhouette=%.4f", k, km.inertia_, score)

    return pd.DataFrame(results)


def main() -> None:
    """
    CLI training entry point.

    Deliberately NOT guarded by `if __name__ == "__main__":` in this
    file. Running this file directly (e.g. `python -m src.zones.kmeans_zones`)
    would execute it as __main__, causing PickupZoneClusterer to be
    pickled with an incorrect module reference - joblib.load() would
    then fail from any other process. Run scripts/train_zones.py instead.
    """
    from src.utils.config_loader import load_config, resolve_path
    from src.visualization.plots import plot_pickup_zones

    logging.basicConfig(level=logging.INFO)

    config = load_config()
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    models_dir = resolve_path(config["paths"]["models_dir"])
    reports_dir = resolve_path(config["paths"]["reports_dir"])

    df = pd.read_parquet(processed_dir / "train_features.parquet")

    clusterer = PickupZoneClusterer(
        n_clusters=config["zones"]["n_clusters"],
        random_state=config["zones"]["random_state"],
    )
    clusterer.fit(df)
    clusterer.save(models_dir / "kmeans_zones.joblib")

    df = assign_pickup_zone(df, clusterer)
    logger.info("Zone size distribution:\n%s", df[ZONE_COLUMN].value_counts().sort_index())

    df.to_parquet(processed_dir / "train_zoned.parquet", index=False)
    logger.info("Saved zoned dataset to %s", processed_dir / "train_zoned.parquet")

    plot_pickup_zones(df, save_path=reports_dir / "pickup_zones.png")