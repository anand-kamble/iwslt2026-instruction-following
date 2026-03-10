"""Configuration loading utilities."""

from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load task config, merging with base config.

    Supports a 'defaults' field that lists base configs to merge.
    Task-specific values override base values.
    """
    config_dir = Path(config_path).parent

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Merge with base config if specified
    defaults = config.pop("defaults", [])
    for default in defaults:
        base_path = config_dir / f"{default}.yaml"
        if base_path.exists():
            with open(base_path) as f:
                base = yaml.safe_load(f)
            # Base values are overridden by task-specific values
            merged = {**base, **config}
            # Deep merge for nested dicts
            for key in base:
                if isinstance(base[key], dict) and key in config and isinstance(config[key], dict):
                    merged[key] = {**base[key], **config[key]}
            config = merged

    return config
