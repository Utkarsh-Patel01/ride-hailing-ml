"""
config_loader.py

Loads and validates the project's central YAML configuration file.

Every other module in this project should obtain configuration values by
importing `load_config` from this module rather than parsing YAML directly.
This guarantees:
  - The config file is read and validated exactly once per process (cached).
  - All required top-level sections are verified to exist before any
    pipeline code runs, so a missing or misspelled key fails immediately
    at startup instead of deep inside a model training run an hour later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Resolves to the project root regardless of the current working directory
# this module happens to be invoked from:
# repo_root/src/utils/config_loader.py -> parents[2] -> repo_root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

# Every top-level section a well-formed config.yaml must define. Adding a
# new section to config.yaml? Add its name here too, so a typo or an
# accidentally deleted section is caught immediately, not silently treated
# as an empty dict deep inside some pipeline.
_REQUIRED_SECTIONS = (
    "project",
    "paths",
    "data",
    "zones",
    "demand_forecast",
    "duration_model",
    "anomaly_detection",
    "api",
)

_config_cache: Optional[Dict[str, Any]] = None


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed, or incomplete."""


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Load and validate the project's YAML configuration.

    Args:
        config_path: Path to the config file. Defaults to
            <project_root>/config/config.yaml if not provided.
        force_reload: If True, bypasses the in-memory cache and re-reads
            the file from disk. Useful in tests that load a temporary
            config fixture.

    Returns:
        The parsed configuration as a nested dictionary.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, does not
            parse into a dictionary, or is missing a required section.
    """
    global _config_cache

    if _config_cache is not None and not force_reload:
        return _config_cache

    resolved_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH

    if not resolved_path.exists():
        raise ConfigError(
            f"Config file not found at '{resolved_path}'. "
            "Did you create config/config.yaml as described in Phase 2?"
        )

    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse '{resolved_path}' as YAML: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(
            f"Config file '{resolved_path}' did not parse into a dictionary "
            f"(got {type(config).__name__}). Check for indentation errors."
        )

    missing_sections = [section for section in _REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        raise ConfigError(
            f"Config file '{resolved_path}' is missing required section(s): "
            f"{missing_sections}. See config/config.yaml from Phase 2 for the full schema."
        )

    logger.info("Loaded configuration from %s", resolved_path)
    _config_cache = config
    return config


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path stored in config.yaml (relative to the project root)
    into an absolute Path object.

    Args:
        relative_path: A path string as stored in config.yaml, e.g.
            'data/raw/train.csv'.

    Returns:
        An absolute Path, so callers never need to worry about their own
        current working directory when a script is invoked.
    """
    return _PROJECT_ROOT / relative_path