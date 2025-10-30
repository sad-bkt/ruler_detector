#!/usr/bin/env python3
"""CLI entrypoint for the EDA pipeline."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.pipelines.eda import run_eda_pipeline


@hydra.main(config_path="../src/configs", config_name="eda", version_base=None)
def main(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd())
    run_eda_pipeline(cfg, original_cwd)


if __name__ == "__main__":
    main()
