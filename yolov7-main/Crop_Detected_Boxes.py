import os
import cv2
from pathlib import Path

# 获取最新的 runs/detect/exp*
detect_dir = Path("runs/detect")
exp_dirs = sorted(
    [p for p in detect_dir.glob("exp*") if p.is_dir() and p.stem.replace("exp", "").isdigit()],
    key=lambda p: int(p.stem.replace("exp", "")),
    reverse=True
)

if not exp_dirs:
    print("❌ No detection result found in runs/detect/")
    exit()

latest_exp = exp_dirs[0]
images_dir = latest_exp
labels_dir = latest_exp / "labels"

if not labels_dir.exists():
    print(f"❌ No labels found in {labels_dir}. Ensure detect.py was run with --save-txt.")
    exit()

# 找到可用的 carsvisionN 文件夹名
base = Path(".")
n = 1
while (base / f"carsvision{n}").exists():
    n += 1
output_root = base / f"carsvision{n}"
output_root.mkdir()
print(f"📂 Created folder: {output_root}")

# 遍历 labels/*.txt
for label_file in labels_dir.glob("*.txt"):
    image_name = label_file.stem + ".jpg"
    image_path = images_dir / image_name
    if not image_path.exists():
        print(f"⚠️ Image not found: {image_name}")
        continue

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    # 每张图对应一个子文件夹
    subfolder = output_root / f"{label_file.stem}_cropped"
    subfolder.mkdir(parents=True, exist_ok=True)

    with open(label_file, "r") as f:
        lines = f.readlines()

    # 提取框并过滤太小的，再按 x_center 排序
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x_c, y_c, bw, bh = map(float, parts[:5])

        # 过滤太小的框（小于30像素）
        if bw * w < 80 or bh * h < 80:
            continue

        boxes.append((x_c, y_c, bw, bh))

    boxes.sort(key=lambda b: b[0])  # 按 x_center 排序

    for idx, (x_c, y_c, bw, bh) in enumerate(boxes, 1):
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        crop = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]

        save_name = f"west{idx}_{label_file.stem}_cropped.jpg"
        save_path = subfolder / save_name
        cv2.imwrite(str(save_path), crop)

print("✅ All crops saved and sorted left-to-right.")
