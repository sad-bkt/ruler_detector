from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class SSDDetector(nn.Module):
    """Обертка над torchvision SSDlite для детекции линеек.

    Интерфейс совместим с существующим тренировочным циклом:
    - в режиме обучения возвращает {"losses": Dict[str, Tensor]}
    - в режиме eval возвращает {"predictions": List[Dict[str, Tensor]]}
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.5,
    ) -> None:
        super().__init__()

        # В torchvision SSDlite класс background включается автоматически,
        # поэтому передаем (num_classes) как количество включая background? Нет —
        # SSDlite ожидает количество foreground классов; background добавляется внутри.
        # Для ruler-кейса: num_classes=2 (background + 1 класс линейки) => здесь укажем 1.
        foreground_classes = max(1, num_classes - 1)

        self.detector = ssdlite320_mobilenet_v3_large(
            weights="DEFAULT" if pretrained_backbone else None,
            num_classes=foreground_classes,
        )

        # Настройки порогов постобработки
        if hasattr(self.detector, "score_thresh"):
            self.detector.score_thresh = score_thresh
        if hasattr(self.detector, "nms_thresh"):
            self.detector.nms_thresh = nms_thresh

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
        **kwargs,
    ) -> dict:
        images = list(images)
        if targets is not None:
            targets = [{k: v for k, v in target.items()} for target in targets]

        if self.training and targets is not None:
            loss_dict = self.detector(images, targets)
            # torchvision SSD возвращает ключи: 'classification', 'bbox_regression'
            return {"losses": loss_dict}

        predictions = self.detector(images)
        return {"predictions": predictions}
