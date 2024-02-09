from ultralytics import YOLO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load a model
model = YOLO('/content/runs/detect/train7/weights/best.pt')

img_path = '/content/307.jpg'  # Replace with the path to your test image
img = Image.open(img_path).convert('RGB')

# Make predictions
results = model(img)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()