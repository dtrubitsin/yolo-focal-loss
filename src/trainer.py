from copy import copy
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

from focal_loss import CustomDetectionLoss


class CustomModel(DetectionModel):
    """Model with FocalLoss integration."""

    def __init__(self, cfg: str = None, ch: int = 3, nc: int = None, verbose: bool = True,
                 gamma: float = 2.0, alpha: float = 0.25):
        super().__init__(cfg, ch, nc, verbose)
        self.gamma = gamma
        self.alpha = alpha

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return CustomDetectionLoss(self, gamma=self.gamma, alpha=self.alpha)


class CustomTrainer(DetectionTrainer):
    """Trainer to run custom versions of YOLO Detection model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        # Get gamma and alpha from overrides
        ovr = dict(overrides or {})  # don't mutate caller's dict
        self.gamma = float(ovr.pop("focal_gamma", 2.0))
        self.alpha = float(ovr.pop("focal_alpha", 0.25))

        super().__init__(cfg, ovr, _callbacks)

        # persist for DDP/resume and run reproducibility
        self.args.focal_gamma = self.gamma
        self.args.focal_alpha = self.alpha

        LOGGER.info(f"CustomTrainer initialized with Focal Loss: γ={self.gamma}, α={self.alpha}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Returns a customized detection model with Focal Loss support."""
        model = CustomModel(
            cfg=cfg or self.model if isinstance(self.model, str) else None,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            gamma=self.gamma,
            alpha=self.alpha
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        args = copy(self.args)

        # Remove unknown keys from cfg
        del args.focal_alpha
        del args.focal_gamma

        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks
        )
