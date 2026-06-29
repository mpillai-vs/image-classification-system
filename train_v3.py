import os
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from copy import deepcopy

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
EPOCHS     = 100
BATCH_SIZE = 64
IMG_SIZE   = 224
WARMUP     = 5       # LR warmup epochs
T          = 6.0     # KD temperature  (higher = softer targets)
ALPHA      = 0.8     # weight on KD loss (0.8 KD + 0.2 CE)
MEAN       = [0.5071, 0.4867, 0.4408]
STD        = [0.2675, 0.2565, 0.2761]
VAL_SPLIT  = 0.1

# ── Transforms ────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Data ──────────────────────────────────────────────────────────
full_train = datasets.CIFAR100("./data", train=True,
                                transform=train_transform, download=True)
val_size   = int(len(full_train) * VAL_SPLIT)
train_size = len(full_train) - val_size
train_subset, val_subset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
val_subset.dataset = datasets.CIFAR100("./data", train=True,
                                        transform=val_transform, download=False)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_subset,   batch_size=128,
                          shuffle=False, num_workers=0)

# ── Models ────────────────────────────────────────────────────────
# Teacher: EfficientNet-B4 pretrained, frozen
teacher = timm.create_model("efficientnet_b4", pretrained=True, num_classes=100)
teacher = teacher.to(DEVICE)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print(f"Teacher parameters : {sum(p.numel() for p in teacher.parameters()):,}")

# Student: EfficientNet-B0 pretrained, to be trained
student = timm.create_model("efficientnet_b0", pretrained=True, num_classes=100)
student = student.to(DEVICE)
total_params = sum(p.numel() for p in student.parameters())
print(f"Student parameters : {total_params:,}")

# EMA of student weights for a better final model
ema_student = deepcopy(student)
ema_student.eval()
EMA_DECAY = 0.9998

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)

# ── CutMix ────────────────────────────────────────────────────────
def cutmix_data(images, labels, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)
    _, _, H, W = images.shape
    cut_h = int(H * (1 - lam) ** 0.5)
    cut_w = int(W * (1 - lam) ** 0.5)
    cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed  = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[idx], lam_adj

# ── KD Loss ───────────────────────────────────────────────────────
def kd_loss(student_logits, teacher_logits, labels, lam=None, labels_b=None):
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits  / T, dim=1)
    kd   = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    if lam is not None:
        ce = (lam * F.cross_entropy(student_logits, labels)
              + (1 - lam) * F.cross_entropy(student_logits, labels_b))
    else:
        ce = F.cross_entropy(student_logits, labels)

    return ALPHA * kd + (1 - ALPHA) * ce

# ── Optimizer + Scheduler ─────────────────────────────────────────
optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

def lr_lambda(epoch):
    if epoch < WARMUP:
        return (epoch + 1) / WARMUP
    progress = (epoch - WARMUP) / (EPOCHS - WARMUP)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── Train loop ────────────────────────────────────────────────────
def train_one_epoch(student, teacher, loader, optimizer, device):
    student.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            t_logits = teacher(images)

        use_cutmix = torch.rand(1).item() > 0.5
        if use_cutmix:
            images, labels_a, labels_b, lam = cutmix_data(images, labels)
            s_logits = student(images)
            loss = kd_loss(s_logits, t_logits, labels_a, lam, labels_b)
        else:
            s_logits = student(images)
            loss = kd_loss(s_logits, t_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        update_ema(ema_student, student, EMA_DECAY)

        total_loss += loss.item()
        _, pred     = s_logits.max(1)
        total      += labels.size(0)
        correct    += pred.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = model(images).max(1)
            total  += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100. * correct / total


# ── Train ─────────────────────────────────────────────────────────
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        student, teacher, train_loader, optimizer, DEVICE)
    val_acc     = evaluate(student,     val_loader, DEVICE)
    val_acc_ema = evaluate(ema_student, val_loader, DEVICE)
    scheduler.step()

    best_this_epoch = max(val_acc, val_acc_ema)
    if best_this_epoch > best_val_acc:
        best_val_acc = best_this_epoch
        # Save whichever (student or EMA) scored higher
        save_model = ema_student if val_acc_ema >= val_acc else student
        torch.save(save_model.state_dict(), "best_model_v3.pth")

    print(f"Epoch [{epoch:03d}/{EPOCHS}] "
          f"Loss: {train_loss:.3f} | "
          f"Train: {train_acc:.1f}% | "
          f"Val: {val_acc:.1f}% | "
          f"EMA Val: {val_acc_ema:.1f}%")

score = best_val_acc / total_params
print(f"\nDone!  Best val accuracy : {best_val_acc:.2f}%")
print(f"       Total parameters  : {total_params:,}")
print(f"       Contest Score     : {score:.10f}")
