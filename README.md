
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
- Model        : EfficientNet-B0
- Parameters   : 5,288,548
- Top-1 Accuracy: ~76%
- Contest Score : Accuracy / Parameters

## Notes
- Trained on CIFAR-100 train set only (50,000 images)
- Test set never used during training
- MPS (Mac Silicon), CUDA, and CPU all supported
