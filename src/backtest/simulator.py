"""
End-to-end simulation driver (stub).

Loads ``configs/experiment.yaml``, builds datasets, runs enabled models, applies
portfolio rules and costs, and returns a serialisable results dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


class RunSimulation:
    """
    High-level CLI target: load config, run pipeline, collect outputs.

    This class is intentionally minimal until wired to data loaders and trainers.
    """

    def __init__(self, config_path: Union[Path, str]) -> None:
        self.config_path = Path(config_path)

    @staticmethod
    def load_config(path: Path | str) -> Dict[str, Any]:
        """Parse YAML experiment configuration."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline described by the config file.

        Stub implementation: validates YAML load and returns a placeholder dict.
        """
        cfg = self.load_config(self.config_path)
        return {
            "status": "stub",
            "config_path": str(self.config_path),
            "universe_keys": list(cfg.get("universe", {}).keys()),
        }
