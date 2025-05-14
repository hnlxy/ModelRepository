import os
import cv2
from glob import glob
from pathlib import Path

def get_auto_bbox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    H, W = img.shape[:2]
    x_c = (x + w / 2) / W
    y_c = (y + h / 2) / H
    w /= W
    h /= H
    return (x_c, y_c, w, h)

def convert_mvtec_to_yolo(mvtec_root, yolo_root):
    classes = sorted([d for d in os.listdir(mvtec_root) if os.path.isdir(os.path.join(mvtec_root, d))])
    os.makedirs(os.path.join(yolo_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_root, 'labels'), exist_ok=True)

    for class_id, cls in enumerate(classes):
        test_dir = os.path.join(mvtec_root, cls, 'test')
        if not os.path.exists(test_dir):
            continue

        for subcls in os.listdir(test_dir):
            img_paths = glob(os.path.join(test_dir, subcls, '*.png'))
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                save_img_path = os.path.join(yolo_root, 'images', f"{cls}_{Path(img_path).name}")
                save_lbl_path = os.path.join(yolo_root, 'labels', f"{cls}_{Path(img_path).stem}.txt")
                bbox = get_auto_bbox(img)
                if bbox:
                    x_c, y_c, w, h = bbox
                    with open(save_lbl_path, 'w') as f:
                        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                cv2.imwrite(save_img_path, img)

    # 保存类别名称
    with open(os.path.join(yolo_root, 'classes.txt'), 'w') as f:
        for cls in classes:
            f.write(cls + '\n')

# 示例调用
convert_mvtec_to_yolo('C:/yolov7-main/data/MVTec-AD', 'C:/yolov7-main/data/yolo_MVTec')
