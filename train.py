
import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ── Device ────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 64
IMG_SIZE   = 224
MEAN       = [0.5071, 0.4867, 0.4408]
STD        = [0.2675, 0.2565, 0.2761]

# ── Transforms ────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ── Data ──────────────────────────────────────────────────────────
train_dataset = datasets.CIFAR100("./data", train=True,
                                   transform=train_transform, download=True)
test_dataset  = datasets.CIFAR100("./data", train=False,
                                   transform=test_transform,  download=True)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=0, pin_memory=False)
test_loader   = DataLoader(test_dataset,  batch_size=128,
                           shuffle=False, num_workers=0, pin_memory=False)

# ── Model ─────────────────────────────────────────────────────────
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=100)
model = model.to(DEVICE)

# ── Training Setup ────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = model(images).max(1)
        total   += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            _, pred = model(images).max(1)
            total   += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100. * correct / total

# ── Train ─────────────────────────────────────────────────────────
best_accuracy = 0.0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE)
    test_acc = evaluate(model, test_loader, DEVICE)
    scheduler.step()

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch [{epoch:02d}/{EPOCHS}] "
          f"Loss: {train_loss:.3f} | "
          f"Train: {train_acc:.1f}% | "
          f"Test: {test_acc:.1f}%")

print(f"Done! Best accuracy: {best_accuracy:.2f}%")
