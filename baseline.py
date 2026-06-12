from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # auto-downloads pretrained weights

results = model.val(
    data="coco.yaml",
    imgsz=640,
    split="val",
    project="runs/baseline"
)

print(f"Baseline mAP@0.5: {results.box.map50:.4f}")
print(f"Baseline mAP@0.5:0.95: {results.box.map:.4f}")