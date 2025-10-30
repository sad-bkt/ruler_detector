from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class DetectionLoss(nn.Module):
    """Aggregates detection losses produced by the model."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        losses: Optional[Dict[str, torch.Tensor]] = None,
        **batch,
    ) -> Dict[str, torch.Tensor]:
        if losses is None:
            images = batch.get("images")
            device = images[0].device if images else "cpu"
            return {"loss": torch.tensor(0.0, device=device)}

        total_loss = torch.stack([value for value in losses.values()]).sum()
        output = {"loss": total_loss}
        for name, value in losses.items():
            output[f"loss_{name}"] = value
        return output
