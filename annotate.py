#!/usr/bin/env python3
"""CLI entrypoint for the annotation pipeline."""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.pipelines.annotate import run_annotation_pipeline


@hydra.main(config_path="../src/configs", config_name="annotate", version_base=None)
def main(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd())
    run_annotation_pipeline(cfg, original_cwd)


if __name__ == "__main__":
    main()
