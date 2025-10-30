from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

try:
    from pycocotools.coco import COCO
except ImportError as exc:  # pragma: no cover - surfaced to user
    raise ImportError("Install 'pycocotools' to work with COCO datasets.") from exc

from src.utils.io_utils import ROOT_PATH

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class RulerCocoDataset(Dataset):
    images_dir: str
    ann_file: Optional[str] = None
    image_ids_file: Optional[str] = None
    is_train: bool = False
    # Optional external instance transforms (Hydra) applied in __getitem__
    instance_transform_train: Optional[object] = None
    instance_transform_inference: Optional[object] = None
    normalize: bool = True
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        self.images_dir = str(self.images_dir)
        self.images_root = Path(self.images_dir)
        assert self.images_root.exists(), f"Images dir not found: {self.images_root}"

        self.has_annotations = self.ann_file is not None

        if self.has_annotations:
            ann_path = Path(self.ann_file)
            assert ann_path.exists(), f"Annotation file not found: {ann_path}"
            self.coco = COCO(str(ann_path))
            self.image_ids = self._load_ids()
        else:
            self.coco = None
            self.image_ids = self._gather_image_paths()

        if self.is_train:
            random.shuffle(self.image_ids)

    def _load_ids(self) -> List[int]:
        if self.image_ids_file:
            ids_path = Path(self.image_ids_file)
            assert ids_path.exists(), f"Image ids file not found: {ids_path}"
            with ids_path.open("r", encoding="utf-8") as fp:
                ids = [int(line.strip()) for line in fp if line.strip()]
            return ids
        return sorted(self.coco.getImgIds())

    def _gather_image_paths(self) -> List[str]:
        return sorted(
            str(path)
            for path in self.images_root.rglob("*")
            if path.suffix.lower() in ALLOWED_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, image_path: Path) -> Image.Image:
        with Image.open(image_path) as img:
            return img.convert("RGB")

    def _resolve_image_path(self, file_name: str) -> Path:
        candidate = Path(file_name)
        if candidate.is_absolute():
            return candidate

        parts = list(candidate.parts)
        if self.images_root.name in parts:
            idx = parts.index(self.images_root.name)
            candidate = Path(*parts[idx + 1 :])
        elif parts and parts[0] == "data":
            # drop leading project-level folder if present
            try:
                idx = parts.index(self.images_root.name)
                candidate = Path(*parts[idx + 1 :])
            except ValueError:
                candidate = Path(*parts[1:])

        resolved = (self.images_root / candidate).resolve()
        if resolved.exists():
            return resolved

        from_root = (ROOT_PATH / candidate).resolve()
        if from_root.exists():
            return from_root

        fallback = (self.images_root / candidate.name).resolve()
        return fallback

    def _load_targets(
        self, image_id: int, width: int, height: int
    ) -> Dict[str, torch.Tensor]:
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes: List[List[float]] = []
        areas: List[float] = []
        iscrowd: List[int] = []
        labels: List[int] = []

        for ann in annotations:
            if "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            areas.append(float(w * h))
            iscrowd.append(int(ann.get("iscrowd", 0)))
            labels.append(1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
            "size": torch.tensor([height, width], dtype=torch.int64),
        }
        return target

    # Legacy in-dataset augmentations removed in favor of instance transforms

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.has_annotations:
            image_id = self.image_ids[index]
            image_info = self.coco.loadImgs([image_id])[0]
            image_path = self._resolve_image_path(image_info["file_name"])
            assert image_path.exists(), f"Image not found: {image_path}"
        else:
            image_path = Path(self.image_ids[index])
            image_id = index

        image = self._load_image(image_path)
        width, height = image.size

        image_tensor = F.to_tensor(image)

        if self.has_annotations:
            target = self._load_targets(image_id, width=width, height=height)
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([image_id]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "size": torch.tensor([height, width], dtype=torch.int64),
            }

        # Apply external instance transforms (flip/rotate) if provided
        if self.is_train and self.instance_transform_train is not None:
            image_tensor, target = self.instance_transform_train(image_tensor, target)
        elif (not self.is_train) and self.instance_transform_inference is not None:
            image_tensor, target = self.instance_transform_inference(
                image_tensor, target
            )
            new_height = int(target["size"][0].item())
            new_width = int(target["size"][1].item())
            if target["boxes"].numel() > 0:
                boxes = target["boxes"]
                boxes[:, ::2] = boxes[:, ::2].clamp(0, float(new_width))
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(new_height))
                target["boxes"] = boxes

        # Normalize after transforms
        if self.normalize:
            image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)

        return {
            "image": image_tensor,
            "target": target,
            "image_id": image_id,
            "path": str(image_path),
        }
