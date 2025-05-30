from ultralytics import YOLO

model = YOLO("yolov8s-pose.pt")  # 自动下载并加载模型
results = model("your_image.jpg")
results[0].plot()  # 可视化结果
