
import os
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ── Device ────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408],
                         [0.2675, 0.2565, 0.2761])
])
test_dataset = datasets.CIFAR100("./data", train=False,
                                  download=True, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=128,
                          shuffle=False, num_workers=0)

# ── Load Model ────────────────────────────────────────────────────
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=100)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("✅ Model loaded from best_model.pth")

# ── Evaluate ──────────────────────────────────────────────────────
correct, total = 0, 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        _, predicted   = model(images).max(1)
        total         += labels.size(0)
        correct       += predicted.eq(labels).sum().item()

# ── Score ─────────────────────────────────────────────────────────
accuracy     = 100. * correct / total
total_params = sum(p.numel() for p in model.parameters())
score        = accuracy / total_params

print()
print("=" * 50)
print(f"  Top-1 Accuracy   : {accuracy:.2f}%")
print(f"  Total Parameters : {total_params:,}")
print(f"  Contest Score    : {score:.10f}")
print("=" * 50)
