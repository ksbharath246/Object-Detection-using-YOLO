from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data=r"C:\Users\ksbha\Documents\Yolo\data.yaml", epochs=10, imgsz=640)
results = model.train(data='/content/data.yaml', epochs=50, imgsz=640)

