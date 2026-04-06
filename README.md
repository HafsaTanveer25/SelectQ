# SelectQ Reproduction

This repository reproduces the main idea of **SelectQ: Calibration Data Selection for Post-training Quantization**.

## What it does

1. Loads a pretrained CNN from torchvision.
2. Extracts activation mean/std layer-wise.
3. Learns SelectQ centroids by dynamic clustering.
4. Ranks training images using knowledge distance.
5. Selects top-K calibration images.
6. Runs simple PTQ with min-max asymmetric quantization.
7. Evaluates Top-1 / Top-5 accuracy.

## Dataset layout

This code expects ImageNet-style folders:

```text
DATA_ROOT/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Run

```bash
python main.py --config configs/resnet18_imagenet.yaml
```

## Notes

- This implementation is intentionally modular.
- It is a faithful engineering reproduction of the paper idea, not a guaranteed line-by-line reimplementation of the authors' private code.
- Start from ResNet18 before extending to other models.
- After reproducing SelectQ, connect your CAT fitting code inside `selectq/pipeline.py`.

## Suggested first experiments

- Random baseline, ResNet18, W4A4
- SelectQ, ResNet18, W4A4
- Random baseline, ResNet18, W6A6
- SelectQ, ResNet18, W6A6
- Random baseline, ResNet18, W8A8
- SelectQ, ResNet18, W8A8
