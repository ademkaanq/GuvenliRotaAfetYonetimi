# =========================================================
# RescueNet – Fine-tune (Yol ağırlıklı)
# Mevcut modeli yükler, CE loss + yüksek road weight ile devam eder
# =========================================================

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# -----------------------------
# Paths & Parameters
# -----------------------------
TILED_TRAIN_IMG = Path("data/tiled/train/images")
TILED_TRAIN_MSK = Path("data/tiled/train/masks")
TILED_VAL_IMG   = Path("data/tiled/val/images")
TILED_VAL_MSK   = Path("data/tiled/val/masks")

NUM_CLASSES        = 12
IMG_SIZE           = 512
BATCH_SIZE         = 7
ACCUMULATION_STEPS = 4
EPOCHS             = 8
LR                 = 3e-5  # Düşük LR — fine-tune için
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Yol (8) ağırlığı yüksek
CLASS_WEIGHTS = torch.tensor([
    1.5,   # Background
    1.5,   # Debris
    1.5,   # Water
    1.0,   # Building_No_Damage
    1.5,   # Building_Minor_Damage
    1.5,   # Building_Major_Damage
    2.0,   # Building_Total_Destruction
    2.0,   # Vehicle
    6.0,   # Road ← yüksek!
    1.5,   # Tree
    2.0,   # Pool
    1.5,   # Sand
], dtype=torch.float).to(DEVICE)

# -----------------------------
# Transforms
# -----------------------------
train_tfms = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.Normalize(),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# -----------------------------
# Dataset
# -----------------------------
class RescueNetDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path  = self.imgs[idx]
        mask_path = self.mask_dir / f"{img_path.stem}_lab.png"
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        aug = self.transform(image=image, mask=mask)
        return aug["image"], aug["mask"].long()

# -----------------------------
# Model — mevcut checkpoint'i yükle
# -----------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None,
).to(DEVICE)

model.load_state_dict(torch.load("rescuenet_best.pth", map_location=DEVICE))
print("✅ rescuenet_best.pth yüklendi")

# -----------------------------
# Loss — Dice + Focal + CE (road ağırlıklı)
# -----------------------------
dice_loss  = smp.losses.DiceLoss(mode="multiclass", smooth=1.0)
focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=3.0, alpha=0.25)
ce_loss    = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

def loss_fn(pred, target):
    return dice_loss(pred, target) + focal_loss(pred, target) + ce_loss(pred, target)

# -----------------------------
# Dataset & Loader
# -----------------------------
train_ds = RescueNetDataset(TILED_TRAIN_IMG, TILED_TRAIN_MSK, train_tfms)
val_ds   = RescueNetDataset(TILED_VAL_IMG,   TILED_VAL_MSK,   val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)} tile")
print(f"🖥️  Device: {DEVICE.upper()}\n")

# -----------------------------
# Optimizer & Scheduler
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# -----------------------------
# Training Loop
# -----------------------------
best_val_loss = float("inf")

print("=" * 55)
print("Fine-tune başlıyor...")
print("=" * 55)

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y) / ACCUMULATION_STEPS
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_val = loss.item() * ACCUMULATION_STEPS
        if loss_val == loss_val:
            total_loss += loss_val

    train_loss = total_loss / len(train_loader)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_loss += loss_fn(model(x), y).item()
    val_loss /= len(val_loader)

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "rescuenet_road.pth")
        print(f"  💾 Best model kaydedildi (val_loss={val_loss:.4f})")

print("\n✅ Fine-tune tamamlandı!")
print(f"   Model: rescuenet_road.pth")