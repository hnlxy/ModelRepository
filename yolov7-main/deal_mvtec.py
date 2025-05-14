import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path

def mask_to_yolo(mask_path, image_size):
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        print(f"Warning: Failed to load mask: {mask_path}")
        return []

    # 明确二值化处理
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        x_c = (x + w / 2) / image_size[0]
        y_c = (y + h / 2) / image_size[1]
        w /= image_size[0]
        h /= image_size[1]
        bboxes.append((0, x_c, y_c, w, h))  # class_id = 0
    return bboxes


def convert_dataset(mvtec_root, yolo_root, target_class='screw'):
    image_dir = f"{mvtec_root}/{target_class}/test"
    mask_dir = f"{mvtec_root}/{target_class}/ground_truth"
    save_img_dir = f"{yolo_root}/images"
    save_lbl_dir = f"{yolo_root}/labels"
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    for subcls in os.listdir(image_dir):
        if subcls == 'good':  # ✅ 跳过无缺陷图像
            continue

        img_paths = glob(f"{image_dir}/{subcls}/*.png")
        for img_path in img_paths:
            filename = Path(img_path).name
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            yolo_lbl_path = os.path.join(save_lbl_dir, filename.replace('.png', '.txt'))
            mask_path = f"{mask_dir}/{subcls}/{filename.replace('.png', '_mask.png')}"

            if os.path.exists(mask_path):
                bboxes = mask_to_yolo(mask_path, (w, h))
                print(f"{mask_path} -> Detected {len(bboxes)} bbox(es)")
                with open(yolo_lbl_path, 'w') as f:
                    for bbox in bboxes:
                        f.write(' '.join(map(str, bbox)) + '\n')
            else:
                print(f"[WARN] Missing mask for: {img_path}")

            cv2.imwrite(os.path.join(save_img_dir, filename), img)

# 执行转换
for cls in [ 'toothbrush','screw']:
    convert_dataset('C:/yolov7-main/data/MVTec-AD', 'C:/yolov7-main/data/yolo_MVTec', cls)
