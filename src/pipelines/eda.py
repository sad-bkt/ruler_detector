from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.logger import setup_logging
from src.pipelines.common import ensure_directory, resolve_path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"]


def build_runtime_config(cfg: DictConfig, root: Path) -> Dict[str, Any]:
    """Resolve all configurable paths and defaults relative to the project root."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    dataset = resolve_path(cfg_dict.get("dataset", "data/ruler/unlabeled"), root)
    output_dir = resolve_path(cfg_dict.get("output_dir", "outputs/eda"), root)

    save_cfg = cfg_dict.get("save", {}) or {}
    save_excel = resolve_path(save_cfg.get("excel"), root)
    save_json = resolve_path(save_cfg.get("json"), root)

    anomalies_cfg = cfg_dict.get("anomalies", {}) or {}
    area_cfg = anomalies_cfg.get("area", {}) or {}
    anomalies_runtime_cfg = {
        "enabled": anomalies_cfg.get("enabled", True),
        "max_entries": anomalies_cfg.get("max_entries", 12),
        "area": {
            "iqr_multiplier": float(area_cfg.get("iqr_multiplier", 1.5)),
            "min": area_cfg.get("min"),
            "max": area_cfg.get("max"),
        },
    }

    return {
        "dataset": dataset,
        "extensions": tuple(
            ext.lower() for ext in cfg_dict.get("extensions", tuple(IMAGE_EXTENSIONS))
        ),
        "limit": cfg_dict.get("limit"),
        "output_dir": output_dir,
        "save_excel": save_excel,
        "save_json": save_json,
        "anomalies": anomalies_runtime_cfg,
    }


def iter_image_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    """Collect all image files with supported extensions under the dataset root."""
    extensions = {ext.lower() for ext in extensions}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)


def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file to detect duplicates."""
    digest = hashlib.md5()
    with path.open("rb") as fp:
        while chunk := fp.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_records(
    image_paths: Sequence[Path],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Open images and collect per-file statistics (resolution, hashes, sizes)."""
    records: List[Dict[str, Any]] = []
    iterator = image_paths if limit is None else image_paths[:limit]
    for path in tqdm(iterator, desc="Scanning images"):
        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                format_name = (img.format or path.suffix.replace(".", "")).upper()
                channels = len(img.getbands())
        except UnidentifiedImageError:
            tqdm.write(f"[warning] Unable to open image: {path}")
            continue

        file_size_bytes = path.stat().st_size
        file_size_kb = file_size_bytes / 1024
        aspect_ratio = width / height if height else None
        area = width * height
        digest = compute_md5(path)
        records.append(
            {
                "path": str(path),
                "filename": path.name,
                "format": format_name,
                "mode": mode,
                "channels": channels,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "area": area,
                "file_size_kb": file_size_kb,
                "file_size_bytes": file_size_bytes,
                "md5": digest,
            }
        )
    return records


def detect_size_anomalies(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Detect unusually small/large images using IQR thresholds or explicit bounds."""
    if not cfg.get("enabled", True) or df.empty:
        return {}

    if "area" not in df:
        return {}

    area_series = df["area"].dropna()
    if area_series.empty:
        return {}

    area_cfg = cfg.get("area", {}) or {}
    iqr_multiplier = float(area_cfg.get("iqr_multiplier", 1.5))
    max_entries = int(cfg.get("max_entries", 12) or 0)

    q1 = float(area_series.quantile(0.25))
    q3 = float(area_series.quantile(0.75))
    iqr = q3 - q1

    min_area = area_cfg.get("min")
    max_area = area_cfg.get("max")

    source = "config"
    if min_area is None or max_area is None:
        source = "iqr"
        if iqr <= 0:
            # All images share identical area; no anomaly detection possible.
            return {}
        if min_area is None:
            min_area = max(q1 - iqr_multiplier * iqr, 0.0)
        if max_area is None:
            max_area = q3 + iqr_multiplier * iqr

    min_area = float(min_area)
    max_area = float(max_area)

    small_df = df[df["area"] <= min_area]
    large_df = df[df["area"] >= max_area]

    if max_entries > 0:
        if not small_df.empty:
            small_df = small_df.nsmallest(max_entries, "area")
        if not large_df.empty:
            large_df = large_df.nlargest(max_entries, "area")

    def to_records(sub_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if sub_df.empty:
            return []
        columns = [
            "path",
            "width",
            "height",
            "area",
            "file_size_kb",
            "aspect_ratio",
            "mode",
        ]
        available_columns = [col for col in columns if col in sub_df.columns]
        records = sub_df[available_columns].copy()
        if "file_size_kb" in records:
            records["file_size_kb"] = records["file_size_kb"].map(
                lambda value: float(value) if pd.notna(value) else value
            )
        records["area"] = records["area"].map(lambda value: float(value))
        return records.to_dict(orient="records")

    return {
        "criteria": {
            "source": source,
            "area_min": min_area,
            "area_max": max_area,
            "iqr_multiplier": iqr_multiplier,
            "q1": q1,
            "q3": q3,
        },
        "small_images": to_records(small_df),
        "large_images": to_records(large_df),
    }


def summarize_dataframe(
    df: pd.DataFrame, anomaly_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Aggregate descriptive statistics, aspect ratios, duplicates, and anomalies."""
    if df.empty:
        return {}

    numeric_columns = [
        column
        for column in ("width", "height", "aspect_ratio", "area", "file_size_kb")
        if column in df
    ]
    numeric_summary = (
        df[numeric_columns].describe().to_dict() if numeric_columns else {}
    )

    per_extension = Counter(Path(row).suffix.lower() for row in df["filename"].tolist())
    per_format = Counter(df["format"].tolist())
    per_mode = Counter(df["mode"].tolist())
    per_channels = Counter(df["channels"].tolist())

    aspect_ratios = [r for r in df["aspect_ratio"].dropna().tolist()]
    bucket_edges = [0.5, 0.75, 1.0, 1.33, 1.78, 2.0, 3.0]
    buckets = defaultdict(int)
    for ratio in aspect_ratios:
        for edge in bucket_edges:
            if ratio <= edge:
                buckets[f"<= {edge:.2f}"] += 1
                break
        else:
            buckets["> 3.00"] += 1

    resolution_buckets = defaultdict(int)
    for _, row in df.iterrows():
        shorter_side = min(row["width"], row["height"])
        if shorter_side < 512:
            resolution_buckets["<512"] += 1
        elif shorter_side < 1024:
            resolution_buckets["512-1023"] += 1
        elif shorter_side < 2048:
            resolution_buckets["1024-2047"] += 1
        else:
            resolution_buckets["2048+"] += 1

    duplicates = {
        digest: group["path"].tolist()
        for digest, group in df.groupby("md5")
        if len(group) > 1
    }

    file_size_series = df.get("file_size_kb")
    file_size_stats = (
        {
            "mean_kb": float(file_size_series.mean()),
            "min_kb": float(file_size_series.min()),
            "max_kb": float(file_size_series.max()),
        }
        if file_size_series is not None and not file_size_series.empty
        else {}
    )

    anomalies = detect_size_anomalies(df, anomaly_cfg)

    return {
        "num_images": int(len(df)),
        "unique_formats": len(per_format),
        "formats": dict(sorted(per_format.items(), key=lambda x: x[1], reverse=True)),
        "extensions": dict(
            sorted(per_extension.items(), key=lambda x: x[1], reverse=True)
        ),
        "modes": dict(sorted(per_mode.items(), key=lambda x: x[1], reverse=True)),
        "channels": dict(sorted(per_channels.items(), key=lambda x: x[0])),
        "numeric_summary": numeric_summary,
        "aspect_ratio_buckets": dict(buckets),
        "resolution_buckets": dict(resolution_buckets),
        "duplicate_groups": duplicates,
        "file_size_kb_stats": file_size_stats,
        "anomalies": anomalies,
    }


def log_summary(logger: logging.Logger, summary: Dict[str, Any]) -> None:
    """Emit human-readable summary of dataset metrics to the logger."""
    if not summary:
        logger.warning("No valid images found in dataset.")
        return

    logger.info("Total images: %s", summary["num_images"])
    logger.info("Extensions: %s", summary["extensions"])  # выводим только расширения
    logger.info("Aspect ratio buckets: %s", summary["aspect_ratio_buckets"])
    logger.info("Resolution buckets: %s", summary["resolution_buckets"])
    if summary["duplicate_groups"]:
        logger.info(
            "Potential duplicates detected: %d groups", len(summary["duplicate_groups"])
        )
    else:
        logger.info("No duplicate images detected.")
    if summary.get("file_size_kb_stats"):
        logger.info("File size stats (KB): %s", summary["file_size_kb_stats"])
    anomalies = summary.get("anomalies") or {}
    if anomalies:
        logger.info(
            "Anomalies detected: small=%d, large=%d (area thresholds %.1f..%.1f)",
            len(anomalies.get("small_images", [])),
            len(anomalies.get("large_images", [])),
            anomalies.get("criteria", {}).get("area_min", float("nan")),
            anomalies.get("criteria", {}).get("area_max", float("nan")),
        )


def run_eda_pipeline(
    cfg: DictConfig,
    root: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Top-level entry point for dataset EDA."""
    logger = logger or logging.getLogger("pipeline.eda")
    runtime_cfg = build_runtime_config(cfg, root)

    dataset: Path = runtime_cfg["dataset"]
    output_dir: Path = runtime_cfg["output_dir"]
    save_excel: Optional[Path] = runtime_cfg["save_excel"]
    save_json: Optional[Path] = runtime_cfg["save_json"]

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger.info("Starting EDA pipeline with dataset=%s", dataset)

    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    if not dataset or not dataset.exists():
        logger.error("Dataset path does not exist: %s", dataset)
        return summary, records

    image_paths = iter_image_files(dataset, runtime_cfg["extensions"])
    if not image_paths:
        logger.warning(
            "No images found under %s with extensions %s",
            dataset,
            runtime_cfg["extensions"],
        )
        return summary, records

    records = collect_records(image_paths, runtime_cfg["limit"])
    df = pd.DataFrame(records) if records else pd.DataFrame()
    summary = (
        summarize_dataframe(df, runtime_cfg.get("anomalies", {}))
        if not df.empty
        else {}
    )
    log_summary(logger, summary)

    ensure_directory(save_excel)
    ensure_directory(save_json)
    ensure_directory(output_dir)

    if save_excel and not df.empty:
        df.to_excel(save_excel, index=False)
        logger.info("Saved per-image statistics to %s", save_excel)

    if save_json and summary:
        import json

        save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Saved summary to %s", save_json)

    return summary, records
