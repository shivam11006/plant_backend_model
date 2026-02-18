from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import json
import os

app = Flask(__name__)

# Load Model Configuration
num_classes = 3
class_indices_path = 'class_indices.json'
recommendations_path = 'recommendations.json'
model_path = 'plant_disease_model.pth'

# Initialize Model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Global variables
class_names = []
recommendations = {}

def load_resources():
    global class_names, recommendations
    try:
        # Load Model
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        else:
            print("Model file not found. Please train the model first.")

        # Load Class Mapping
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_to_idx = json.load(f)
                class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
                print(f"Class names loaded: {class_names}")
        else:
            print(f"Warning: {class_indices_path} not found.")

        # Load Recommendations
        if os.path.exists(recommendations_path):
            with open(recommendations_path, 'r') as f:
                recommendations = json.load(f)
                print("Recommendations loaded.")
        else:
            print(f"Warning: {recommendations_path} not found.")
    except Exception as e:
        print(f"Error loading resources: {e}")

# Load resources at startup
load_resources()

def is_leaf_like(image_pil):
    """
    Checks if the image has plant-like characteristics (Color + Texture).
    """
    img_np = np.array(image_pil.convert('RGB'))
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

    # Color Heuristic
    greenish = (g > r) & (g > b) & (g > 40)
    yellowish_brownish = (r > b) & (g > b) & (r > 50) & (g > 40)
    plant_pixels = np.sum(greenish | yellowish_brownish)
    pixel_count = img_np.shape[0] * img_np.shape[1]

    # Texture Heuristic
    gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
    grad_x = np.diff(gray, axis=1)
    grad_y = np.diff(gray, axis=0)
    texture_score = (np.std(grad_x) + np.std(grad_y)) / 2

    is_textured = (3.0 < texture_score < 35.0)
    is_colored = (plant_pixels / pixel_count > 0.15)

    return is_textured and is_colored

def get_prediction(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Physical verification
    if not is_leaf_like(image_pil):
        return "Not a Leaf", 0.0, 0.0

    image_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        top_probs, top_classes = torch.topk(probabilities, 2, dim=1)
        confidence = top_probs[0][0].item()
        margin = confidence - top_probs[0][1].item()

        if class_names:
            return class_names[top_classes[0][0].item()], confidence, margin

        return "Unknown", 0.0, 0.0

@app.route('/')
def home():
    return jsonify({"message": "Plant Disease Detection API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_bytes = file.read()
        disease_name, confidence, margin = get_prediction(img_bytes)

        # Statistical verification
        if disease_name == "Not a Leaf" or confidence < 0.98 or margin < 0.8:
            return jsonify({
                'error': 'This object is not recognized as a leaf.',
                'advice': 'Please upload a clear image of a plant leaf.',
                'details': {
                    'detected_as': disease_name if disease_name != "Not a Leaf" else "Unknown Object",
                    'confidence': round(confidence, 2)
                }
            })

        rec = recommendations.get(disease_name, {
            'disease': disease_name,
            'pesticide': 'No specific pesticide recommended.',
            'treatment': 'Consult a local agricultural expert.'
        })

        return jsonify({
            'disease': rec.get('disease', disease_name),
            'pesticide': rec.get('pesticide', 'N/A'),
            'fertilizer_or_treatment': rec.get('treatment', 'N/A'),
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
