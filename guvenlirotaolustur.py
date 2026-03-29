import cv2
import numpy as np
import heapq
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


BW_PATH = Path(r"saferoute\new.jpg")

print("Tam yol:", BW_PATH.resolve())
print("Dosya var mı:", BW_PATH.exists())
bw_image = cv2.imread(str(BW_PATH), cv2.IMREAD_GRAYSCALE)


H, W       = bw_image.shape
CAR_RADIUS = max(4, int(min(H, W) * 0.015))

# ── Erozyon: dar geçitleri kapat ──────────────────────────
kernel    = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (CAR_RADIUS * 2 + 1, CAR_RADIUS * 2 + 1)
)
binary    = (bw_image >= 128).astype(np.uint8) * 255
eroded    = cv2.erode(binary, kernel)
safe_mask = (eroded >= 128).astype(np.float32)

def get_points_from_user(bw_image):
    points = []
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(bw_image, cmap='gray')
    ax.set_title("Başlangıç noktasına tıkla. Beyaz yollar, Siyah Engeller olmalı", fontsize=13)
    ax.axis("off")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        points.append((y, x))
        if len(points) == 1:
            ax.plot(x, y, 'go', markersize=14)
            ax.set_title("Bitiş noktasına tıkla", fontsize=13)
            fig.canvas.draw()
        elif len(points) == 2:
            ax.plot(x, y, 'ro', markersize=14)
            ax.set_title("Hesaplanıyor...", fontsize=13)
            fig.canvas.draw()
            plt.pause(0.3)
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()
    return points[0], points[1]


def find_safe_route(safe_mask, start, end):
    H, W = safe_mask.shape
    cost = np.where(safe_mask >= 0.5, 1.0, np.inf).astype(np.float32)

    dist = np.full((H, W), np.inf)
    dist[start] = 0
    prev = {}
    heap = [(0.0, start)]

    while heap:
        d, (y, x) = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        if (y, x) == end:
            break
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and np.isfinite(cost[ny, nx]):
                nd = d + cost[ny, nx]
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(heap, (nd, (ny, nx)))


    path, node = [], end
    while node in prev:
        path.append(node)
        node = prev[node]
    path.append(start)
    path.reverse()
    return path


def draw_route_bw(bw_image, path, thickness=4):
    result = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGB)
    for i in range(1, len(path)):
        y1, x1 = path[i-1]
        y2, x2 = path[i]
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    cv2.circle(result, (path[0][1],  path[0][0]),  12, (0, 255, 0), -1)
    cv2.circle(result, (path[-1][1], path[-1][0]), 12, (255, 0, 0), -1)
    return result


start, end = get_points_from_user(bw_image)
path = find_safe_route(safe_mask, start, end)

route_img = draw_route_bw(bw_image, path)

out_path = BW_PATH.stem.replace("_segmentation_bw", "") + "_route.png"
cv2.imwrite(out_path, cv2.cvtColor(route_img, cv2.COLOR_RGB2BGR))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].imshow(bw_image, cmap='gray'); axes[0].set_title("S/B Maske");    axes[0].axis("off")
axes[1].imshow(route_img);             axes[1].set_title("Güvenli Rota"); axes[1].axis("off")
plt.suptitle(f"{BW_PATH.name}", fontsize=14)
plt.tight_layout()
plt.show()