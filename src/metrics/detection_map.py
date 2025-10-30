from __future__ import annotations

from typing import Any, Dict, List

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.metrics.base_metric import BaseMetric


class DetectionMeanAPMetric(BaseMetric):
    """Wrapper around torchmetrics MeanAveragePrecision for detection models."""

    def __init__(self, device: str = "auto", name: str | None = None, **kwargs: Any):
        super().__init__(name=name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = MeanAveragePrecision(**kwargs).to(device)
        self.device = device

    def __call__(
        self,
        predictions: List[Dict[str, torch.Tensor]] | None = None,
        targets: List[Dict[str, torch.Tensor]] | None = None,
        **batch: Any,
    ) -> float:
        if predictions is None:
            predictions = batch.get("predictions")
        if targets is None:
            targets = batch.get("targets")

        if not predictions or targets is None:
            return 0.0

        preds_cpu = []
        for pred in predictions:
            preds_cpu.append(
                {
                    "boxes": pred.get("boxes", torch.empty((0, 4))).to(self.device),
                    "scores": pred.get("scores", torch.empty((0,))).to(self.device),
                    "labels": pred.get(
                        "labels", torch.empty((0,), dtype=torch.int64)
                    ).to(self.device),
                }
            )

        targets_cpu = []
        for tgt in targets:
            targets_cpu.append(
                {
                    "boxes": tgt.get("boxes", torch.empty((0, 4))).to(self.device),
                    "labels": tgt.get(
                        "labels", torch.empty((0,), dtype=torch.int64)
                    ).to(self.device),
                }
            )

        self.metric.update(preds_cpu, targets_cpu)
        result = self.metric.compute()
        map_val = result.get("map", torch.tensor(0.0, device=self.device)).item()
        self.metric.reset()
        return map_val
