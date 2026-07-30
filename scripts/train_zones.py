"""
train_zones.py

Thin CLI entry point for Phase 6's pickup-zone clustering training.

Run this file directly - NOT `python -m src.zones.kmeans_zones` - so
that src.zones.kmeans_zones is always imported normally rather than
executed as __main__. PickupZoneClusterer is defined in that file;
pickling it while that file is __main__ bakes an incorrect module
reference into kmeans_zones.joblib, breaking joblib.load() from any
other process (see this phase's debugging note for the full story).
"""

from src.zones.kmeans_zones import main

if __name__ == "__main__":
    main()