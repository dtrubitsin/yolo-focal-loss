"""Custom trainer and model implementations for YOLO with Focal Loss.

This module provides custom implementations of YOLO detection models and trainers
that integrate Focal Loss for improved handling of class imbalance in object
detection tasks.
"""

from copy import copy
from typing import Any

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

from focal_loss import CustomDetectionLoss


class CustomModel(DetectionModel):
    """Custom YOLO detection model with Focal Loss integration.

    This model extends the standard DetectionModel to support Focal Loss
    parameters (gamma and alpha) for better handling of class imbalance.

    Attributes:
        gamma (float): Controls how much the model "suppresses" easy examples.
            Higher values focus more on hard examples.
        alpha (float): Sets the weight for positive examples, helping combat
            class imbalance.
    """

    def __init__(
        self,
        cfg: str = None,
        ch: int = 3,
        nc: int = None,
        verbose: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ):
        """Initialize the custom detection model with Focal Loss parameters.

        Args:
            cfg (str, optional): Model configuration file path. Defaults to None.
            ch (int, optional): Number of input channels. Defaults to 3.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            gamma (float, optional): Focusing parameter for Focal Loss. Higher
                values focus more on hard examples. Defaults to 2.0.
            alpha (float, optional): Weighting factor for positive examples.
                Helps combat class imbalance. Defaults to 0.25.
        """
        super().__init__(cfg, ch, nc, verbose)
        self.gamma = gamma
        self.alpha = alpha

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel.

        Returns:
            CustomDetectionLoss: A custom detection loss instance with
                the configured gamma and alpha parameters.
        """
        return CustomDetectionLoss(self, gamma=self.gamma, alpha=self.alpha)


class CustomTrainer(DetectionTrainer):
    """Custom trainer for YOLO detection models with Focal Loss support.

    This trainer extends the standard DetectionTrainer to support Focal Loss
    parameters and automatically configures the model with the appropriate
    loss function.

    Attributes:
        gamma (float): Focusing parameter for Focal Loss.
        alpha (float): Weighting factor for positive examples.
    """

    def __init__(
        self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None
    ):
        """Initialize the custom trainer with Focal Loss parameters.

        Args:
            cfg: Training configuration. Defaults to DEFAULT_CFG.
            overrides (dict[str, Any] | None, optional): Configuration overrides.
                Can include custom keys, so 'focal_gamma' and 'focal_alpha' removed.
                Defaults to None.
            _callbacks: Optional callbacks for the trainer. Defaults to None.
        """
        # Get gamma and alpha from overrides
        ovr = dict(overrides or {})  # don't mutate caller's dict
        self.gamma = float(ovr.pop("focal_gamma", 2.0))
        self.alpha = float(ovr.pop("focal_alpha", 0.25))

        super().__init__(cfg, ovr, _callbacks)

        # persist for DDP/resume and run reproducibility
        self.args.focal_gamma = self.gamma
        self.args.focal_alpha = self.alpha

        LOGGER.info(
            f"CustomTrainer initialized with Focal Loss: γ={self.gamma}, α={self.alpha}"
        )

    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ):
        """Get a customized detection model with Focal Loss support.

        Args:
            cfg (str | None, optional): Model configuration file path.
                Defaults to None.
            weights (str | None, optional): Path to model weights file.
                Defaults to None.
            verbose (bool, optional): Whether to print verbose output.
                Defaults to True.

        Returns:
            CustomModel: A custom detection model instance configured with
                Focal Loss parameters.
        """
        model = CustomModel(
            cfg=cfg or self.model if isinstance(self.model, str) else None,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            gamma=self.gamma,
            alpha=self.alpha,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Get a DetectionValidator for YOLO model validation.

        This method creates a validator instance and removes Focal Loss specific
        parameters from the configuration to ensure compatibility with the
        standard cfg validation process.

        Returns:
            DetectionValidator: A detection validator instance configured for
                the current test data loader and training arguments.
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        args = copy(self.args)

        # Remove unknown keys from cfg
        del args.focal_alpha
        del args.focal_gamma

        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=args,
            _callbacks=self.callbacks,
        )
