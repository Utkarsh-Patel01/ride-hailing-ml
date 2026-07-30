"""
train_anomaly_detector.py

Thin CLI entry point for Phase 11's anomaly detector training.
Run this file directly - NOT `python -m src.models.anomaly.train_isolation_forest`
- for the same __main__-pickling reason documented in scripts/train_zones.py.
"""

from src.models.anomaly.train_isolation_forest import main

if __name__ == "__main__":
    main()
    