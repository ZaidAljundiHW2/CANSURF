from ultralytics import YOLO

model = YOLO('yolo11s.pt')

results = model.train(

    data='CANSURF/data.yaml',
    epochs=150,
    imgsz=640,
    patience=15,
    batch=64,
    
)