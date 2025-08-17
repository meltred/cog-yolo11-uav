from ultralytics import YOLO

model = YOLO("/workspace/runs/detect/train/weights/best.pt")
metrics = model.val(
    data="/workspace/UAV-3/data.yaml",
    split="val",
    imgsz=640,
    batch=16,
    device=0,
    plots=True
)

print("Results dictionary:", metrics)
print("mAP50-95:", metrics.box.map)
print("mAP50:", metrics.box.map50)
print("Per-class mAPs:", metrics.box.maps)


# 0: dj-air3 (96.8% accuracy) - 1: uav (80.1% accuracy)