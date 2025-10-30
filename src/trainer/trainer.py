import torch
from PIL import Image, ImageDraw

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        if (
            not self.is_train
            and getattr(self, "compute_eval_loss", False)
            and "targets" in batch
        ):
            raw_losses = outputs.get("losses")
            if raw_losses is None:
                was_training = self.model.training
                self.model.train()
                with torch.set_grad_enabled(False):
                    loss_outputs = self.model(
                        images=batch["images"], targets=batch["targets"]
                    )
                self.model.train(mode=was_training)
                if isinstance(loss_outputs, dict):
                    raw_losses = loss_outputs.get("losses", loss_outputs)
            if isinstance(raw_losses, dict):
                batch["losses"] = raw_losses

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if self.writer is None:
            return

        images = batch.get("images")
        if not images:
            return

        # During training the detector returns only losses; skip logging in that case.
        if mode == "train" and not batch.get("predictions"):
            return

        try:
            image_tensor = images[0].detach().cpu()
        except (IndexError, AttributeError):
            return

        if image_tensor.ndim != 3:
            return

        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if image_tensor.shape[0] == 1:
            mean = torch.tensor([0.0]).view(-1, 1, 1)
            std = torch.tensor([1.0]).view(-1, 1, 1)

        img = image_tensor * std + mean
        img = img.clamp(0.0, 1.0)
        img_uint8 = img.permute(1, 2, 0).mul(255).to(torch.uint8).numpy()

        try:
            image = Image.fromarray(img_uint8)
        except Exception:
            return

        draw = ImageDraw.Draw(image)

        targets = batch.get("targets")
        if targets:
            gt_boxes = targets[0].get("boxes")
            if isinstance(gt_boxes, torch.Tensor) and gt_boxes.numel() > 0:
                for box in gt_boxes.cpu().tolist():
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    label_y = y1 - 12 if y1 - 12 > 0 else y1 + 4
                    draw.text((x1 + 4, label_y), "gt", fill=(255, 0, 0))

        predictions = batch.get("predictions")
        if predictions:
            pred = predictions[0]
            boxes = pred.get("boxes")
            scores = pred.get("scores")
            if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                boxes_list = boxes.cpu().tolist()
                scores_list = (
                    scores.cpu().tolist() if isinstance(scores, torch.Tensor) else []
                )
                for idx, box in enumerate(boxes_list):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                    if scores_list:
                        score = scores_list[idx]
                        label_y = y1 - 12 if y1 - 12 > 0 else y1 + 4
                        draw.text(
                            (x1 + 4, label_y),
                            f"pred {score:.2f}",
                            fill=(0, 255, 0),
                        )

        self.writer.add_image("detections", image)
