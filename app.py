import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import gradio as gr
import os

# Configuration
num_classes = 3
class_indices_path = 'class_indices.json'
recommendations_path = 'recommendations.json'
model_path = 'plant_disease_model.pth'

# Load model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

class_names = []
recommendations = {}

def load_resources():
    global class_names, recommendations

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_to_idx = json.load(f)
            class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]

    if os.path.exists(recommendations_path):
        with open(recommendations_path, 'r') as f:
            recommendations = json.load(f)

load_resources()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    disease = class_names[pred.item()]
    rec = recommendations.get(disease, {})

    result = {
        "disease": disease,
        "confidence": round(confidence.item(), 2)
    }
    result.update(rec) # Add all recommendation fields (pesticide, treatment, severity, etc.)
    return result

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="json",
    title="Plant Disease Detection"
)

interface.launch()
