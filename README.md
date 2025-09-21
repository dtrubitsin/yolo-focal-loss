# YOLOv8/YOLOv11 with Focal Loss

This repository provides a **custom training extension for Ultralytics YOLOv8/YOLOv11** that replaces the standard BCE
loss with **Focal Loss** for classification.  
The integration is seamless and allows better handling of class imbalance in object detection tasks.

---

## 📌 Features

- Drop-in replacement of the standard YOLO detection loss with **Focal Loss**.
- Fully compatible with Ultralytics training pipeline (`yolo train ...`).
- Configurable `gamma` and `alpha` parameters via training overrides.
- Custom `Trainer` and `Model` classes for easy integration.
- DDP and resume options supported.

---

## 📂 Project structure

```
yolo-focal-loss/
├── src/
│   ├── __init__.py                 
│   ├── focal_loss.py               # CustomDetectionLoss: YOLOv8/11 loss with Focal Loss
│   └── trainer.py                  # CustomTrainer & CustomModel with Focal Loss support
├── requirements.txt                # Requirements (ultralytics)
├── .gitignore                      # Ignore files (__pycache__, .env)
└── README.md                       # Project documentation
```

---

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Tested on the specified versions, for other versions see changes in original documentation.

### 2. Import and train with `CustomTrainer`

```python
from ultralytics import YOLO

from src.trainer import CustomTrainer

# Example training run with Focal Loss
model = YOLO("yolo11n.pt")
model.train(trainer=CustomTrainer, data="coco8.yaml", epochs=3, focal_gamma=1.5, focal_alpha=0.75)
```

---

## ⚙️ How it works

* `CustomDetectionLoss` inherits from Ultralytics' `v8DetectionLoss`.
* The **classification loss** is replaced with `FocalLoss`, while **box** and **DFL** losses remain unchanged:
    * Crete target tensor with zeros;
    * Place foreground target scores;
    * Call FocalLoss from original Ultralytics implementation;
    * Divide by `target_scores_sum` to normalize.

```python
if self.nc > 1:  # cls loss (only if multiple classes)
    # Prepare targets for focal loss
    t = torch.zeros_like(pred_scores, dtype=dtype, device=self.device)
    if fg_mask.sum():
        t[fg_mask] = target_scores[fg_mask].to(dtype)

    # Use focal loss instead BCE
    loss[1] = self.focal_loss(pred_scores, t) / target_scores_sum
```

* `CustomModel` ensures proper initialization of the loss with the given parameters.
  Extend `DetectionModel` with gamma and alpha for FocalLoss initialization from yaml configuration.
* `CustomTrainer` plugs into the Ultralytics training pipeline, managing overrides and validation.
  Extend `DetectionTrainer` with proper parameters initialization which support overrides from arguments or yaml config
  and DDP/resume and run reproducibility.

---

## 📊 Why Focal Loss?

Standard BCE loss can struggle with **class imbalance** (many easy negatives vs. few positives).
Focal Loss down-weights easy examples and focuses training on hard ones, which often improves:

* Rare class detection
* Small object detection
* Robustness to imbalance in datasets

---

## 🛠 Contributing

This project demonstrates how to **extend Ultralytics YOLO** by customizing its training pipeline.
It can be used as a template for integrating other loss functions or research ideas.

Pull requests and discussions are welcome!

---

## 📜 References

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [Focal Loss paper](https://arxiv.org/abs/1708.02002)
* [Ultralytics FocalLoss reference](https://docs.ultralytics.com/reference/utils/loss/#ultralytics.utils.loss.FocalLoss)

---
