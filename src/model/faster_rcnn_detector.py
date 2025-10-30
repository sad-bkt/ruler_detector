from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNDetector(nn.Module):
    """Wrapper around torchvision Faster R-CNN for object detection."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        score_thresh: float = 0.25,
        nms_thresh: float = 0.5,
    ) -> None:
        super().__init__()

        self.detector = fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained_backbone else None,
            trainable_backbone_layers=trainable_backbone_layers,
            box_score_thresh=score_thresh,
            box_nms_thresh=nms_thresh,
        )

        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

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
            return {"losses": loss_dict}

        predictions = self.detector(images)
        return {"predictions": predictions}
