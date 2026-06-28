
# CIFAR-100 Image Classification
## Model: EfficientNet-B0 | Transfer Learning

## Setup
```bash
pip install -r requirements.txt
```

## Training (on Train data only)
```bash
python train.py
```

## Inference / Evaluation
```bash
python infer.py
```

## Results

| Metric            | Value          |
|-------------------|----------------|
| Model             | EfficientNet-B0 |
| Total Parameters  | 4,135,648      |
| Top-1 Accuracy    | 85.44%         |
| Contest Score     | 0.0000206594   |

> Contest Score = Top-1 Accuracy / Total Parameters

## Notes
- Trained on CIFAR-100 train set only (50,000 images)
- Test set never used during training
- MPS (Mac Silicon), CUDA, and CPU all supported
