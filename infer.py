
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
correct_top1, correct_top5, total = 0, 0, 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs        = model(images)

        # Top-1: highest scoring class matches label
        _, pred_top1   = outputs.max(1)
        correct_top1  += pred_top1.eq(labels).sum().item()

        # Top-5: true label is among the 5 highest scoring classes
        _, pred_top5   = outputs.topk(5, dim=1)
        correct_top5  += pred_top5.eq(labels.view(-1, 1)).any(dim=1).sum().item()

        total += labels.size(0)

# ── Score ─────────────────────────────────────────────────────────
top1_acc     = 100. * correct_top1 / total
top5_acc     = 100. * correct_top5 / total
total_params = sum(p.numel() for p in model.parameters())
score        = top1_acc / total_params

print()
print("=" * 50)
print(f"  Top-1 Accuracy   : {top1_acc:.2f}%")
print(f"  Top-5 Accuracy   : {top5_acc:.2f}%")
print(f"  Total Parameters : {total_params:,}")
print(f"  Contest Score    : {score:.10f}")
print("=" * 50)
