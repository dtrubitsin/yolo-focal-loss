"""Custom detection loss implementation with Focal Loss for YOLO models.

This module provides a custom detection loss class that replaces the standard
Binary Cross Entropy (BCE) loss with Focal Loss for better handling of class
imbalance in object detection tasks.
"""

from typing import Any

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import FocalLoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class CustomDetectionLoss(v8DetectionLoss):
    """Custom loss function for YOLOv8/11 with Focal Loss.

    This class inherits from the original v8DetectionLoss and replaces the
    Binary Cross Entropy (BCE) loss with Focal Loss for classification.
    Focal Loss helps address class imbalance by down-weighting easy examples
    and focusing on hard examples.

    Attributes:
        focal_loss (FocalLoss): The focal loss instance for classification.
    """

    def __init__(self, model, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize the custom detection loss with Focal Loss.

        Args:
            model: The detection model instance.
            gamma (float, optional): Focusing parameter for Focal Loss. Higher
                values focus more on hard examples. Defaults to 1.5.
            alpha (float, optional): Weighting factor for positive examples.
                Helps combat class imbalance. Defaults to 0.25.
        """
        super().__init__(model)
        # Replace BCE with Focal Loss for classification
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        LOGGER.info(f"FocalLoss initialized: γ={gamma}, α={alpha}")

    def __call__(
        self, preds: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss for box, classification, and DFL components.

        This method computes the detection loss using Focal Loss for classification
        instead of the standard BCE loss. The loss consists of three components:
        box regression loss, classification loss (using Focal Loss), and DFL loss.

        Args:
            preds: Model predictions, either as a tuple or tensor containing
                feature maps and predictions.
            batch: Dictionary containing batch data with keys:
                - batch_idx: Batch indices
                - cls: Class labels
                - bboxes: Bounding box coordinates

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Total loss multiplied by batch size
                - Detached loss tensor with individual components (box, cls, dfl)
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        # b, grids, 4
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss - use FOCAL LOSS
        if self.nc > 1:  # cls loss (only if multiple classes)
            # Prepare targets for focal loss
            t = torch.zeros_like(pred_scores, dtype=dtype, device=self.device)
            if fg_mask.sum():
                t[fg_mask] = target_scores[fg_mask].to(dtype)

            # Use focal loss instead BCE
            loss[1] = self.focal_loss(pred_scores, t) / target_scores_sum

        # box loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
