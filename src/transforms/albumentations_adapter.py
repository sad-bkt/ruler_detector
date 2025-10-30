from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.transforms import functional as F


class AlbuDetectionAdapter:
    """
    Обёртка над Albumentations.Compose для детекции.

    Ожидает тензор изображения формата (C, H, W) [0..1] и target c ключами:
      - boxes: FloatTensor[N, 4] в формате xyxy
      - labels: LongTensor[N]

    Возвращает преобразованные image, target с обновлёнными bboxes и размером.
    """

    def __init__(self, albu, label_field_name: str = "labels") -> None:
        self.albu = albu
        self.label_field_name = label_field_name

    @staticmethod
    def _tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
        # image_tensor: (C, H, W), float [0..1]
        image_np = (image_tensor.clamp(0, 1) * 255.0).byte().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))  # -> (H, W, C)
        return image_np

    @staticmethod
    def _uint8_image_to_tensor(image_np: np.ndarray) -> torch.Tensor:
        # image_np: (H, W, C), uint8
        image_np = np.ascontiguousarray(image_np)
        image_tensor = F.to_tensor(image_np)  # -> float [0..1], (C, H, W)
        return image_tensor

    def __call__(
        self, image: torch.Tensor, target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_np = self._tensor_to_uint8_image(image)

        boxes: torch.Tensor = target.get(
            "boxes", torch.zeros((0, 4), dtype=torch.float32)
        )
        labels: torch.Tensor = target.get(
            self.label_field_name,
            target.get("labels", torch.zeros((0,), dtype=torch.int64)),
        )

        if boxes.numel() == 0:
            bboxes_list: List[List[float]] = []
            labels_list: List[int] = []
        else:
            bboxes_list = boxes.tolist()  # xyxy для Albumentations c format=pascal_voc
            labels_list = labels.tolist()

        augmented = self.albu(
            image=img_np,
            bboxes=bboxes_list,
            **{self.label_field_name: labels_list},
        )

        img_np_aug = augmented["image"]
        bboxes_aug: List[List[float]] = augmented.get("bboxes", [])
        labels_aug: Optional[List[int]] = augmented.get(self.label_field_name)

        image_out = self._uint8_image_to_tensor(img_np_aug)
        h_new, w_new = image_out.shape[1], image_out.shape[2]

        if len(bboxes_aug) == 0:
            boxes_out = torch.zeros((0, 4), dtype=torch.float32)
            labels_out = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_out = torch.tensor(bboxes_aug, dtype=torch.float32)
            labels_out = torch.tensor(labels_aug, dtype=torch.int64)

        target_out = dict(target)
        target_out["boxes"] = boxes_out
        target_out[self.label_field_name] = labels_out
        target_out["size"] = torch.tensor([h_new, w_new], dtype=torch.int64)

        # Клэмп боксов в пределах изображения
        if boxes_out.numel() > 0:
            boxes_clamped = boxes_out.clone()
            boxes_clamped[:, ::2] = boxes_clamped[:, ::2].clamp(0, float(w_new))
            boxes_clamped[:, 1::2] = boxes_clamped[:, 1::2].clamp(0, float(h_new))
            target_out["boxes"] = boxes_clamped

        return image_out, target_out
