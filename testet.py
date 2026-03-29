# =========================================================
# Smoke Test — Bağımsız, train.py import etmez
# =========================================================
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

TILED_TRAIN_IMG = Path("data/tiled/train/images")
TILED_TRAIN_MSK = Path("data/tiled/train/masks")
TILED_VAL_IMG   = Path("data/tiled/val/images")
TILED_VAL_MSK   = Path("data/tiled/val/masks")

NUM_CLASSES        = 12
IMG_SIZE           = 512
BATCH_SIZE         = 2
ACCUMULATION_STEPS = 4
LR                 = 1e-4
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

train_tfms = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])
val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

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

dice_loss  = smp.losses.DiceLoss(mode="multiclass", smooth=1.0)
focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=3.0, alpha=0.25)

def loss_fn(pred, target):
    return dice_loss(pred, target) + focal_loss(pred, target)

train_ds  = RescueNetDataset(TILED_TRAIN_IMG, TILED_TRAIN_MSK, train_tfms)
val_ds    = RescueNetDataset(TILED_VAL_IMG,   TILED_VAL_MSK,   val_tfms)
train_sub = Subset(train_ds, list(range(min(200, len(train_ds)))))
val_sub   = Subset(val_ds,   list(range(min(50,  len(val_ds)))))
train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False)

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=NUM_CLASSES, activation=None).to(DEVICE)
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler    = GradScaler()

print(f"🔥 Smoke test | {len(train_sub)} train / {len(val_sub)} val | {DEVICE.upper()}\n")

total_nan = 0
for epoch in range(3):
    model.train()
    total_loss, nan_count = 0, 0
    optimizer.zero_grad()

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred, y) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        loss_val = loss.item() * ACCUMULATION_STEPS
        if loss_val != loss_val:
            nan_count += 1
            print(f"  ⚠️  NaN! batch {i+1} | pred: {pred.min():.2f}/{pred.max():.2f} | mask: {y.unique().tolist()}")
        else:
            total_loss += loss_val

    train_loss = total_loss / len(train_loader)
    model.eval()
    val_loss = 0
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        val_loss += loss_fn(model(x), y).item()
    val_loss /= len(val_loader)
    total_nan += nan_count
    print(f"Epoch {epoch+1}/3 | Train: {train_loss:.4f} | Val: {val_loss:.4f} | NaN: {nan_count}")

print()
if total_nan == 0:
    print("🎉 NaN yok — ana eğitimi başlatabilirsin!")
else:
    print(f"❌ Toplam {total_nan} NaN batch — sorun devam ediyor.")