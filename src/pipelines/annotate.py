from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLOE

from src.logger import setup_logging
from src.pipelines.common import ensure_directory, resolve_path


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    score: float
    label: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection dataclass to a plain Python dict."""
        return {
            "bbox": self.bbox,
            "score": self.score,
            "label": self.label,
            "source": self.source,
        }


def clamp_bbox_xyxy(bbox: Sequence[float], width: int, height: int) -> List[float]:
    """Clamp XYXY coordinates to image bounds and keep corners ordered."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), width))
    y1 = max(0.0, min(float(y1), height))
    x2 = max(0.0, min(float(x2), width))
    y2 = max(0.0, min(float(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_xyxy(bbox: Sequence[float], width: int, height: int) -> List[float]:
    """Convert XYXY pixels to normalized [0,1] coordinates."""
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox, width, height)
    width = max(width, 1)
    height = max(height, 1)
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def denormalize_xyxy(bbox: Sequence[float], width: int, height: int) -> List[float]:
    """Convert normalized XYXY coordinates back to pixel space."""
    x1, y1, x2, y2 = bbox
    return [
        float(x1 * width),
        float(y1 * height),
        float(x2 * width),
        float(y2 * height),
    ]


def build_runtime_config(cfg: DictConfig, root: Path) -> Dict[str, Any]:
    """Resolve config values to concrete paths and parsed structures."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]

    dataset = resolve_path(cfg_dict.get("dataset", "data/ruler/unlabeled"), root)
    output_dir = resolve_path(
        cfg_dict.get("output_dir", "outputs/eda_annotate/annotations"), root
    )

    label = str(cfg_dict.get("label", "object"))
    prompts_cfg = cfg_dict.get("prompts")
    if prompts_cfg:
        prompt_variants = [str(p) for p in prompts_cfg if p]
        if not prompt_variants:
            prompt_variants = [label]
    else:
        prompt_variants = [label]

    models: List[Dict[str, Any]] = []
    for raw_model in cfg_dict.get("models", []) or []:
        if not raw_model or not raw_model.get("enabled", True):
            continue
        model_cfg = dict(raw_model)
        model_type = model_cfg.get("type")
        if model_type is None:
            model_type = model_cfg.get("name")
        if model_type is None:
            raise ValueError("Each model must specify a 'type' or 'name'.")
        model_cfg["type"] = model_type
        model_cfg["name"] = model_cfg.get("name", model_type)
        for key in ("config_path", "weights"):
            if model_cfg.get(key):
                model_cfg[key] = resolve_path(model_cfg[key], root)
        model_cfg["device_resolved"] = resolve_device(model_cfg.get("device"))
        models.append(model_cfg)

    postprocess_cfg = cfg_dict.get("postprocess", {}) or {}
    postprocess_cfg = {
        "type": postprocess_cfg.get("type", "weighted_boxes_fusion"),
        "params": postprocess_cfg.get("params", {}) or {},
    }

    filters_cfg = cfg_dict.get("filters", {}) or {}
    if "default" in filters_cfg or "per_source" in filters_cfg:
        default_filters = filters_cfg.get("default", {}) or {}
        per_source_filters = filters_cfg.get("per_source", {}) or {}
    else:
        default_filters = filters_cfg
        per_source_filters = {}
    filters_runtime_cfg = {
        "default": default_filters,
        "per_source": per_source_filters,
    }

    viz_cfg = cfg_dict.get("visualization", {}) or {}
    viz_dir = resolve_path(viz_cfg.get("dir"), root) if viz_cfg.get("dir") else None
    visualization_cfg = {
        "save_overlays": viz_cfg.get("save_overlays", False),
        "dir": viz_dir,
        "max_images_to_log": viz_cfg.get("max_images_to_log", 6),
    }

    return {
        "dataset": dataset,
        "output_dir": output_dir,
        "label": label,
        "prompts": prompt_variants,
        "limit": cfg_dict.get("limit"),
        "models": models,
        "postprocess": postprocess_cfg,
        "filters": filters_runtime_cfg,
        "visualization": visualization_cfg,
    }


def _merged_filters(filters: Dict[str, Any], source: Optional[str]) -> Dict[str, Any]:
    """Merge default filter thresholds with per-model overrides for a given source."""
    default_cfg = filters.get("default", {}) or {}
    source_cfg = {}
    if source:
        source_cfg = filters.get("per_source", {}).get(source, {}) or {}
    merged = dict(default_cfg)
    merged.update({k: v for k, v in source_cfg.items() if v is not None})
    return merged


def filter_detections(
    detections: List[Detection],
    filters: Dict[str, Any],
    *,
    width: int,
    height: int,
    source: Optional[str] = None,
) -> List[Detection]:
    """Apply confidence, area, and aspect filters to raw model detections."""
    if not detections:
        return detections

    config = _merged_filters(filters, source)

    min_score = config.get("min_score")
    min_area = config.get("min_area")
    max_area = config.get("max_area")
    min_aspect = config.get("min_aspect_ratio")
    max_aspect = config.get("max_aspect_ratio")
    min_length = config.get("min_length")

    filtered: List[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        if box_w == 0 or box_h == 0:
            continue
        area = box_w * box_h
        longest = max(box_w, box_h)
        aspect = longest / min(box_w, box_h)

        if min_score is not None and det.score < float(min_score):
            continue
        if min_area is not None and area < float(min_area):
            continue
        if max_area is not None and area > float(max_area):
            continue
        if min_aspect is not None and aspect < float(min_aspect):
            continue
        if max_aspect is not None and aspect > float(max_aspect):
            continue
        if min_length is not None and longest < float(min_length):
            continue
        filtered.append(det)
    return filtered


def resolve_device(device_value: Optional[str]) -> str:
    """Resolve device string, preferring CUDA/MPS when available."""
    device = (device_value or "auto").lower()
    if device in {"auto", "default"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_grounding_dino_model(model_cfg: Dict[str, Any]):
    """Load Grounding DINO checkpoint declared in config."""
    from groundingdino.util.inference import load_model as gd_load_model

    if not model_cfg.get("config_path") or not model_cfg.get("weights"):
        raise FileNotFoundError("Grounding DINO requires 'config_path' and 'weights'.")
    config_path = Path(model_cfg["config_path"])
    weights_path = Path(model_cfg["weights"])
    if not config_path.exists():
        raise FileNotFoundError(f"Grounding DINO config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Grounding DINO weights not found: {weights_path}")

    device = model_cfg.get("device_resolved") or resolve_device(model_cfg.get("device"))
    return gd_load_model(
        model_config_path=str(config_path),
        model_checkpoint_path=str(weights_path),
        device=device,
    )


def run_grounding_dino(
    model,
    image_path: Path,
    label: str,
    prompts: Sequence[str],
    model_cfg: Dict[str, Any],
) -> Tuple[List[Detection], Tuple[int, int]]:
    """Run Grounding DINO on a single image and collect detections."""
    from groundingdino.util.inference import load_image as gd_load_image
    from groundingdino.util.inference import predict as gd_predict

    # Allow overriding caption explicitly, otherwise join prompt synonyms.
    caption = model_cfg.get("caption") or ". ".join(prompts)

    image_source, image_tensor = gd_load_image(str(image_path))
    if isinstance(image_source, np.ndarray):
        height, width = image_source.shape[:2]
    else:
        width, height = image_source.size
    device = model_cfg.get("device_resolved") or resolve_device(model_cfg.get("device"))
    boxes_tensor, logits, phrases = gd_predict(
        model=model,
        image=image_tensor,
        caption=caption,
        box_threshold=model_cfg.get("box_threshold", 0.35),
        text_threshold=model_cfg.get("text_threshold", 0.25),
        device=device,
    )
    boxes = boxes_tensor.cpu().numpy()
    scores = logits.cpu().numpy()
    detections: List[Detection] = []
    for box, score, phrase in zip(boxes, scores, phrases):
        bbox = box.tolist()
        if max(bbox) <= 1.5:
            bbox = denormalize_xyxy(bbox, width, height)
        bbox = clamp_bbox_xyxy(bbox, width, height)
        detections.append(
            Detection(
                bbox=bbox,
                score=float(score),
                label=(phrase.strip() or label),
                source=model_cfg["name"],
            )
        )
    return detections, (width, height)


def load_yoloe_model(model_cfg: Dict[str, Any]):
    """Load YOLOE checkpoint and move to requested device."""
    if not model_cfg.get("weights"):
        raise FileNotFoundError("YOLOE requires 'weights'.")
    weights_path = Path(model_cfg["weights"])
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLOE weights not found: {weights_path}")

    model = YOLOE(str(model_cfg["weights"]))
    device = model_cfg.get("device_resolved") or resolve_device(model_cfg.get("device"))
    if device == "mps":
        logging.getLogger("pipeline.annotate").warning(
            "YOLOE text encoder is not supported on MPS; falling back to CPU."
        )
        device = "cpu"
        model_cfg["device_resolved"] = device
    if device:
        model.to(device)
    return model


def run_yoloe(
    model,
    image_path: Path,
    label: str,
    prompts: Sequence[str],
    model_cfg: Dict[str, Any],
) -> Tuple[List[Detection], Tuple[int, int]]:
    """Run YOLOE on a single image with text prompts and collect detections."""
    classes = model_cfg.get("classes") or list(prompts)
    text_embeddings = model_cfg.get("text_embeddings")
    if text_embeddings is None:
        text_embeddings = model.get_text_pe(classes)
    model.set_classes(classes, text_embeddings)

    results = model.predict(
        source=str(image_path),
        conf=model_cfg.get("conf", 0.25),
        iou=model_cfg.get("iou", 0.45),
        verbose=False,
    )
    if not results:
        return [], (0, 0)

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        width, height = result.orig_shape[1], result.orig_shape[0]
        return [], (width, height)

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels_idx = result.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[int(idx)] for idx in labels_idx]
    width, height = result.orig_shape[1], result.orig_shape[0]

    detections: List[Detection] = []
    for bbox, score, label_name in zip(boxes, scores, labels):
        detections.append(
            Detection(
                bbox=clamp_bbox_xyxy(bbox, width, height),
                score=float(score),
                label=label_name or label,
                source=model_cfg["name"],
            )
        )
    return detections, (width, height)


def fuse_detections(
    sources: Iterable[List[Detection]],
    width: int,
    height: int,
    label: str,
    iou_thr: float,
    skip_box_thr: float,
    fused_source: str,
) -> List[Detection]:
    """Combine detections from different models via weighted boxes fusion."""
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as exc:  # pragma: no cover - surfaced to user
        raise ImportError(
            "Install 'ensemble-boxes' to use weighted boxes fusion."
        ) from exc

    boxes_list: List[List[List[float]]] = []
    scores_list: List[List[float]] = []
    labels_list: List[List[int]] = []
    for detections in sources:
        if not detections:
            continue
        boxes_list.append(
            [normalize_xyxy(det.bbox, width, height) for det in detections]
        )
        scores_list.append([det.score for det in detections])
        labels_list.append([1 for _ in detections])

    if not boxes_list:
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    fused: List[Detection] = []
    for bbox, score, _ in zip(fused_boxes, fused_scores, fused_labels):
        fused.append(
            Detection(
                bbox=clamp_bbox_xyxy(
                    denormalize_xyxy(bbox, width, height), width, height
                ),
                score=float(score),
                label=label,
                source=fused_source,
            )
        )
    return fused


def apply_postprocess(
    postprocess_cfg: Dict[str, Any],
    label: str,
    width: int,
    height: int,
    model_outputs: Dict[str, List[Detection]],
) -> List[Detection]:
    """Post-process raw detections according to the configured strategy (e.g., WBF)."""
    post_type = (postprocess_cfg.get("type") or "weighted_boxes_fusion").lower()
    params = postprocess_cfg.get("params", {}) or {}

    if post_type in {"weighted_boxes_fusion", "wbf"}:
        iou_thr = params.get("iou", 0.55)
        skip_box_thr = params.get("skip", 0.001)
        return fuse_detections(
            sources=model_outputs.values(),
            width=width,
            height=height,
            label=label,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            fused_source=postprocess_cfg.get("name", "fused"),
        )

    if post_type in {"none", "concat"}:
        merged: List[Detection] = []
        for detections in model_outputs.values():
            merged.extend(detections)
        return merged

    raise ValueError(f"Unsupported postprocess type: {postprocess_cfg.get('type')}")


def build_coco_structure(
    items: List[Dict[str, Any]],
    category_name: str,
) -> Dict[str, Any]:
    """Convert per-image detections into a COCO JSON structure."""
    coco: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name}],
    }
    annotation_id = 1
    for image_id, item in enumerate(items, start=1):
        coco["images"].append(
            {
                "id": image_id,
                "file_name": item["image_path"],
                "width": item["width"],
                "height": item["height"],
            }
        )
        for detection in item["fused"]:
            x1, y1, x2, y2 = detection["bbox"]
            width_box = max(0.0, x2 - x1)
            height_box = max(0.0, y2 - y1)
            area = width_box * height_box
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x1, y1, width_box, height_box],
                    "area": area,
                    "score": detection["score"],
                    "source": detection["source"],
                }
            )
            annotation_id += 1
    return coco


def create_visualizations(
    image_path: Path,
    outputs: Dict[str, List[Detection]],
    label: str,
) -> Dict[str, Image.Image]:
    """Render detections from all sources onto a single combined image."""
    colors = {
        "grounding_dino": (0, 255, 0),
        "yolo_world": (255, 165, 0),
        "fused": (255, 0, 0),
    }
    base_image = Image.open(image_path).convert("RGB")
    combined = base_image.copy()
    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default()

    legend_order: List[str] = []
    legend_colors: Dict[str, Tuple[int, int, int]] = {}

    for key, detections in outputs.items():
        if not detections:
            continue
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.source, colors.get(key, (0, 128, 255)))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            display_label = det.label or label
            tag_raw = det.source or key
            tag = "fused_boxes" if tag_raw == "fused" else tag_raw
            text = f"{tag}: {display_label} {det.score:.2f}"
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((x1, y1), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:  # Pillow < 10 fallback
                text_w, text_h = draw.textsize(text, font=font)
            text_y = max(0, y1 - text_h)
            draw.rectangle([x1, text_y, x1 + text_w + 4, text_y + text_h], fill=color)
            draw.text((x1 + 2, text_y), text, fill=(0, 0, 0), font=font)
            if tag not in legend_colors:
                legend_order.append(tag)
                legend_colors[tag] = color

    if legend_order:
        padding = 6
        line_height = (
            font.getbbox("A")[3] - font.getbbox("A")[1]
            if hasattr(font, "getbbox")
            else font.getsize("A")[1]
        )
        legend_width = 0
        for tag in legend_order:
            text = f"{tag}"
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
            else:
                text_w, _ = draw.textsize(text, font=font)
            legend_width = max(legend_width, text_w)
        legend_height = (line_height + padding) * len(legend_order) + padding
        legend_x, legend_y = padding, padding
        draw.rectangle(
            [
                legend_x - padding,
                legend_y - padding,
                legend_x + legend_width + 3 * padding,
                legend_y + legend_height,
            ],
            fill=(255, 255, 255, 220),
        )
        for idx, tag in enumerate(legend_order):
            color = legend_colors[tag]
            y = legend_y + idx * (line_height + padding)
            draw.rectangle(
                [legend_x, y, legend_x + padding * 2, y + line_height], fill=color
            )
            draw.text((legend_x + padding * 3, y), tag, fill=(0, 0, 0), font=font)

    return {"combined": combined}


def prepare_models(
    models_cfg: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Instantiate and return all enabled detectors defined in the config."""
    loaded: Dict[str, Dict[str, Any]] = {}
    for model_cfg in models_cfg:
        model_type = model_cfg["type"].lower()
        name = model_cfg["name"]
        if model_type == "grounding_dino":
            model = load_grounding_dino_model(model_cfg)
        elif model_type in {"yolo_world", "yoloe"}:
            model = load_yoloe_model(model_cfg)
        else:
            raise ValueError(f"Unsupported model type: {model_cfg['type']}")
        loaded[name] = {"type": model_type, "model": model, "config": model_cfg}
    return loaded


def run_annotation_pipeline(
    cfg: DictConfig,
    root: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Top-level entry for automatic annotation + fusion."""
    logger = logger or logging.getLogger("pipeline.annotate")
    runtime_cfg = build_runtime_config(cfg, root)

    dataset: Path = runtime_cfg["dataset"]
    output_dir: Path = runtime_cfg["output_dir"]
    label: str = runtime_cfg["label"]
    prompts: Sequence[str] = runtime_cfg["prompts"]
    limit = runtime_cfg["limit"]
    visualization_cfg = runtime_cfg["visualization"]
    filters_cfg = runtime_cfg.get("filters", {}) or {}

    if not dataset or not dataset.exists():
        msg = f"Dataset not found: {dataset}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    if visualization_cfg["save_overlays"]:
        viz_dir = visualization_cfg["dir"] or (output_dir / "visualizations")
        visualization_cfg["dir"] = viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger.info("Starting annotation pipeline with dataset=%s", dataset)

    image_paths = [
        path
        for path in sorted(dataset.rglob("*"))
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    ]
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise RuntimeError(f"No images with supported extensions found under {dataset}")

    models = prepare_models(runtime_cfg["models"])

    combined_results: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for image_path in tqdm(image_paths, desc="Annotating images"):
        per_model_outputs: Dict[str, List[Detection]] = {}
        width: Optional[int] = None
        height: Optional[int] = None

        for name, model_bundle in models.items():
            model_type = model_bundle["type"]
            model = model_bundle["model"]
            model_cfg = model_bundle["config"]
            if model_type == "grounding_dino":
                detections, (w, h) = run_grounding_dino(
                    model=model,
                    image_path=image_path,
                    label=label,
                    prompts=prompts,
                    model_cfg=model_cfg,
                )
            elif model_type in {"yolo_world", "yoloe"}:
                detections, (w, h) = run_yoloe(
                    model=model,
                    image_path=image_path,
                    label=label,
                    prompts=prompts,
                    model_cfg=model_cfg,
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported model type: {model_type}")
            filtered = filter_detections(
                detections,
                filters_cfg,
                width=w,
                height=h,
                source=name,
            )
            per_model_outputs[name] = filtered
            if width is None or height is None:
                width, height = w, h
            else:
                width = width or w
                height = height or h

        if width is None or height is None:
            raise RuntimeError(f"Failed to determine image size for {image_path}")

        fused_detections = apply_postprocess(
            postprocess_cfg=runtime_cfg["postprocess"],
            label=label,
            width=width,
            height=height,
            model_outputs=per_model_outputs,
        )
        fused_detections = filter_detections(
            fused_detections,
            filters_cfg,
            width=width,
            height=height,
            source="fused",
        )

        record = {
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "label": label,
            "models": {
                name: [det.to_dict() for det in detections]
                for name, detections in per_model_outputs.items()
            },
            "fused": [det.to_dict() for det in fused_detections],
        }
        combined_results.append(record)

        summary_rows.append(
            {
                "image_path": str(image_path),
                **{
                    f"{name}_detections": len(dets)
                    for name, dets in per_model_outputs.items()
                },
                "fused_detections": len(fused_detections),
            }
        )

        if visualization_cfg["save_overlays"]:
            outputs_for_viz = dict(per_model_outputs)
            if len(per_model_outputs) > 1 and fused_detections:
                outputs_for_viz["fused"] = fused_detections
            visuals = create_visualizations(
                image_path=image_path, outputs=outputs_for_viz, label=label
            )
            for key, image in visuals.items():
                ensure_directory(
                    visualization_cfg["dir"] / f"{image_path.stem}_{key}.png"
                )
                image.save(visualization_cfg["dir"] / f"{image_path.stem}_{key}.png")

    raw_json_path = output_dir / "raw_predictions.json"
    coco_path = output_dir / "annotations_coco.json"

    raw_json_path.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
    coco = build_coco_structure(combined_results, category_name=label)
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    logger.info("Saved raw predictions to %s", raw_json_path)
    logger.info("Saved COCO annotations to %s", coco_path)
    if visualization_cfg["save_overlays"]:
        logger.info("Saved visualizations to %s", visualization_cfg["dir"])

    return {
        "raw_predictions_path": raw_json_path,
        "coco_annotations_path": coco_path,
        "num_images": len(combined_results),
    }
