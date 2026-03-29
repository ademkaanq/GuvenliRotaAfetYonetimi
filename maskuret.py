# =========================================================
# RescueNet – 1. Adım: Safe Mask Üretici
# Modelden segmentasyon çıktısı alır, safe_mask'ı .npy
# olarak "masks/" klasörüne kaydeder.
# =========================================================
import torch
import cv2
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Ayarlar ───────────────────────────────────────────────
NUM_CLASSES  = 12
IMG_SIZE     = 512
TILE_OVERLAP = 64
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Background", "Debris", "Water",
    "Building_No_Damage", "Building_Minor_Damage",
    "Building_Major_Damage", "Building_Total_Destruction",
    "Vehicle", "Road", "Tree", "Pool", "Sand"
]

COLORS = [
    [0,   0,   0  ], [128, 64,  0  ], [0,   0,   255],
    [0,   255, 0  ], [255, 255, 0  ], [255, 165, 0  ],
    [255, 0,   0  ], [255, 0,   255], [128, 128, 128],
    [0,   128, 0  ], [0,   255, 255], [255, 228, 196],
]

ROAD_IDX          = 8
VEHICLE_IDX       = 7
BUILDING_MINOR_IDX  = 4
BUILDING_MAJOR_IDX  = 5
BUILDING_TOTAL_IDX  = 6

INPUT_DIR  = Path("saferoute\map")        # işlenecek .jpg'ler
OUTPUT_DIR = Path("saferoute\masks")      # .npy + görsel çıktılar
OUTPUT_DIR.mkdir(exist_ok=True)

val_tfms = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

# ── Model ─────────────────────────────────────────────────
model = smp.Unet(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=NUM_CLASSES, activation=None
).to(DEVICE)
model.load_state_dict(torch.load(r"saferoute\rescuenet_road.pth", map_location=DEVICE))
model.eval()
print(f"✅ Model yüklendi | Device: {DEVICE.upper()}")

# ── Yardımcı Fonksiyonlar ─────────────────────────────────
def predict_large_image(image):
    H, W   = image.shape[:2]
    stride = IMG_SIZE - TILE_OVERLAP
    pred_sum  = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, H - IMG_SIZE + 1, stride):
            for x in range(0, W - IMG_SIZE + 1, stride):
                tile = image[y:y+IMG_SIZE, x:x+IMG_SIZE]
                inp  = val_tfms(image=tile)["image"].unsqueeze(0).to(DEVICE)
                out  = torch.softmax(model(inp), dim=1).squeeze().cpu().float().numpy()
                pred_sum[:, y:y+IMG_SIZE, x:x+IMG_SIZE] += out
                count_map[y:y+IMG_SIZE, x:x+IMG_SIZE]   += 1
    count_map = np.maximum(count_map, 1)
    return (pred_sum / count_map).argmax(axis=0).astype(np.uint8)


def expand_road(mask, expand_px=10):
    road_mask = (mask == ROAD_IDX).astype(np.uint8)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px*2+1, expand_px*2+1))
    mask[cv2.dilate(road_mask, kernel) > 0] = ROAD_IDX
    return mask


def create_safe_mask(mask, danger_radius=15):
    H, W = mask.shape
    safe = np.zeros((H, W), dtype=np.float32)
    safe[mask == ROAD_IDX]    = 1.0
    safe[mask == VEHICLE_IDX] = 1.0
    for bidx in [BUILDING_MINOR_IDX, BUILDING_MAJOR_IDX, BUILDING_TOTAL_IDX]:
        for y, x in zip(*np.where(mask == bidx)):
            cv2.circle(safe, (x, y), danger_radius, 0.6, -1)
    return safe


def colorize_mask(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(COLORS):
        out[mask == i] = c
    return out


def binary_safe_visual(mask):
    """Geçilebilir piksel (yol + araç) = beyaz, engel = siyah."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask == ROAD_IDX]    = 255
    out[mask == VEHICLE_IDX] = 255
    return out


def print_class_distribution(mask, name):
    total = mask.size
    print(f"\n=== {name} ===")
    for i, n in enumerate(CLASS_NAMES):
        c = (mask == i).sum()
        if c > 0:
            print(f"  {n}: {c/total*100:.1f}%")


# ── Ana Döngü ─────────────────────────────────────────────
img_paths = sorted(INPUT_DIR.glob("*.jpg"))
if not img_paths:
    print("⚠️  'map/' klasöründe .jpg bulunamadı!")

for img_path in img_paths:
    #if img_path.name != "test_real.jpg":
     #   continue
    print(f"\n🔍 İşleniyor: {img_path.name}")
    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    if image is None:
        print(f"⚠️  Okunamadı: {img_path}")
        continue

    pred_mask = predict_large_image(image)
    pred_mask = expand_road(pred_mask, expand_px=10)
    print_class_distribution(pred_mask, img_path.name)

    safe_mask = create_safe_mask(pred_mask, danger_radius=15)

    stem = img_path.stem

    # 1) safe_mask  → .npy  (2. script bunu okuyacak)
    npy_path = OUTPUT_DIR / f"{stem}_safe_mask.npy"
    np.save(str(npy_path), safe_mask)
    print(f"  💾 Safe mask kaydedildi : {npy_path}")

    # 2a) pred_mask → renkli PNG
    color_path = OUTPUT_DIR / f"{stem}_segmentation_color.png"
    cv2.imwrite(str(color_path), cv2.cvtColor(colorize_mask(pred_mask), cv2.COLOR_RGB2BGR))
    print(f"  🖼️  Renkli segmentasyon  : {color_path}")

    # 2b) pred_mask → siyah-beyaz PNG  (beyaz=geçilebilir, siyah=engel)
    bw_path = OUTPUT_DIR / f"{stem}_segmentation_bw.png"
    cv2.imwrite(str(bw_path), binary_safe_visual(pred_mask))
    print(f"  🖼️  S/B segmentasyon     : {bw_path}")

print("\n✅ Tüm görüntüler işlendi! Maskeler 'masks/' klasöründe.")