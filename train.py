
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


RAW_TRAIN_IMG  = Path("data/train/train-org-img")
RAW_TRAIN_MSK  = Path("data/train/train-label-img")
RAW_VAL_IMG    = Path("data/val/val-org-img")
RAW_VAL_MSK    = Path("data/val/val-label-img")

TILED_TRAIN_IMG = Path("data/tiled/train/images")
TILED_TRAIN_MSK = Path("data/tiled/train/masks")
TILED_VAL_IMG   = Path("data/tiled/val/images")
TILED_VAL_MSK   = Path("data/tiled/val/masks")

NUM_CLASSES        = 12
IMG_SIZE           = 512
TILE_OVERLAP       = 64
MAX_TILES_PER_IMAGE = 12
BATCH_SIZE         = 9
ACCUMULATION_STEPS = 4    # Efektif batch size = 2 x 4 = 8
EPOCHS             = 20
LR                 = 1e-4
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Background", "Debris", "Water", "Building_No_Damage",
    "Building_Minor_Damage", "Building_Major_Damage",
    "Building_Total_Destruction", "Vehicle", "Road", "Tree", "Pool", "Sand"
]


def tile_dataset(img_dir, mask_dir, out_img_dir, out_mask_dir,
                 tile_size=IMG_SIZE, overlap=TILE_OVERLAP):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    stride = tile_size - overlap

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_paths:
        print(f"⚠️  {img_dir} içinde görüntü bulunamadı.")
        return

    total_tiles = 0
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        mask_path = mask_dir / f"{img_path.stem}_lab.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        H, W = img.shape[:2]

        candidates = []
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                mask_tile    = mask[y:y+tile_size, x:x+tile_size]
                unique_classes = np.unique(mask_tile)
                if len(unique_classes) == 1 and unique_classes[0] == 0:
                    continue
                candidates.append((len(unique_classes), y, x))

        candidates.sort(reverse=True)
        selected = candidates[:MAX_TILES_PER_IMAGE]

        for tile_idx, (_, y, x) in enumerate(selected):
            img_tile  = img[y:y+tile_size, x:x+tile_size]
            mask_tile = mask[y:y+tile_size, x:x+tile_size]
            name = f"{img_path.stem}_t{tile_idx:04d}"
            cv2.imwrite(str(out_img_dir / f"{name}.jpg"), img_tile)
            cv2.imwrite(str(out_mask_dir / f"{name}_lab.png"), mask_tile)

        total_tiles += len(selected)
        print(f" {img_path.name} → {len(selected)} tile ({len(candidates)} adaydan)")

    print(f"\n Toplam {total_tiles} tile oluşturuldu → {out_img_dir}")


def clear_tiles():
    import shutil
    for d in [TILED_TRAIN_IMG, TILED_TRAIN_MSK, TILED_VAL_IMG, TILED_VAL_MSK]:
        if d.exists():
            shutil.rmtree(d)


def prepare_tiles(force=False):
    if force:
        clear_tiles()

    if not TILED_TRAIN_IMG.exists() or not any(TILED_TRAIN_IMG.glob("*.jpg")):
        print("Train görüntüleri tile'lanıyor...")
        tile_dataset(RAW_TRAIN_IMG, RAW_TRAIN_MSK, TILED_TRAIN_IMG, TILED_TRAIN_MSK)
    else:
        print(f"Train tile'ları mevcut ({len(list(TILED_TRAIN_IMG.glob('*.jpg')))} adet)")

    if not TILED_VAL_IMG.exists() or not any(TILED_VAL_IMG.glob("*.jpg")):
        print("Val görüntüleri tile'lanıyor...")
        tile_dataset(RAW_VAL_IMG, RAW_VAL_MSK, TILED_VAL_IMG, TILED_VAL_MSK)
    else:
        print(f"Val tile'ları mevcut ({len(list(TILED_VAL_IMG.glob('*.jpg')))} adet)")



train_tfms = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.GaussNoise(p=0.2),
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

        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()



def build_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    ).to(DEVICE)


def build_loss_fn():
    dice_loss  = smp.losses.DiceLoss(mode="multiclass", smooth=1.0)
    focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=3.0, alpha=0.25)

    def loss_fn(pred, target):
        return dice_loss(pred, target) + focal_loss(pred, target)

    return loss_fn



def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        pred = model(x)
        loss = loss_fn(pred, y) / ACCUMULATION_STEPS
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_val = loss.item() * ACCUMULATION_STEPS
        if loss_val == loss_val:  # NaN değilse ekle
            total_loss += loss_val

    return total_loss / len(loader)


def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()

    return total_loss / len(loader)



def predict_large_image(model, image_path, tile_size=IMG_SIZE, overlap=TILE_OVERLAP):
    image  = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    H, W   = image.shape[:2]
    stride = tile_size - overlap

    pred_sum  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                tile = image[y:y+tile_size, x:x+tile_size]
                inp  = val_tfms(image=tile)["image"].unsqueeze(0).to(DEVICE)
                out  = torch.softmax(model(inp), dim=1).squeeze().cpu().float().numpy()
                pred_sum[:, y:y+tile_size, x:x+tile_size] += out
                count_map[y:y+tile_size, x:x+tile_size]   += 1

    count_map = np.maximum(count_map, 1)
    return (pred_sum / count_map).argmax(axis=0).astype(np.uint8)


if __name__ == "__main__":
    prepare_tiles(force=False)

    train_ds = RescueNetDataset(TILED_TRAIN_IMG, TILED_TRAIN_MSK, train_tfms)
    val_ds   = RescueNetDataset(TILED_VAL_IMG,   TILED_VAL_MSK,   val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"\n📊 Train: {len(train_ds)} tile | Val: {len(val_ds)} tile")
    print(f"🖥️  Device: {DEVICE.upper()}")

    model     = build_model()
    loss_fn   = build_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for param in model.encoder.parameters():
        param.requires_grad = False

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        if epoch == 5:
            print("\nEncoder açıldı (fine-tuning başlıyor).")
            for param in model.encoder.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] = LR * 0.1

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss   = validate(model, val_loader, loss_fn)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "rescuenet_best.pth")
            print(f"Best model kaydedildi (val_loss={val_loss:.4f})")

    torch.save(model.state_dict(), "rescuenet_last.pth")
    print("\nEğitim tamamlandı.")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Modeller      : rescuenet_best.pth / rescuenet_last.pth")