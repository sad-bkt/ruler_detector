from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from src.logger import WandBWriter

PathLike = Union[str, Path]


def resolve_path(path: Optional[PathLike], root: Path) -> Optional[Path]:
    """Resolve a path relative to the given root."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    return path


def ensure_directory(path: Optional[Path]) -> None:
    """Create directory for the given path or the path itself."""
    if path is None:
        return
    target = path if path.suffix == "" else path.parent
    target.mkdir(parents=True, exist_ok=True)


def to_serializable(value: Any) -> Any:
    """Convert config values to W&B-friendly representations."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(val) for val in value]
    return value


def config_to_dict(config: Optional[DictConfig | Dict[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def initialize_wandb_writer(
    cfg: Optional[DictConfig | Dict[str, Any]],
    logger: logging.Logger,
    project_config: Dict[str, Any],
) -> Optional[WandBWriter]:
    """Create W&B writer if logging is enabled."""
    cfg_dict = config_to_dict(cfg)
    if not cfg_dict or not cfg_dict.get("enabled", False):
        return None

    try:
        writer = WandBWriter(
            logger=logger,
            project_config=project_config,
            project_name=cfg_dict.get("project_name", "ruler-detector"),
            entity=cfg_dict.get("entity"),
            run_id=cfg_dict.get("run_id"),
            run_name=cfg_dict.get("run_name"),
            mode=cfg_dict.get("mode", "online"),
            save_code=cfg_dict.get("save_code", False),
        )
        if not hasattr(writer, "wandb"):
            logger.warning("W&B writer is not fully initialized; disabling logging.")
            return None
        return writer
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to initialize W&B logging: %s", exc, exc_info=True)
        return None
