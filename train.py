import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import os
import json

def train_model():
    # Configuration
    data_dir = os.path.join('..', 'data')
    train_dir = os.path.join(data_dir, 'Train', 'Train')
    val_dir = os.path.join(data_dir, 'Validation', 'Validation')
    
    model_save_path = 'plant_disease_model.pth'
    class_indices_path = 'class_indices.json'
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_classes = 3 # Healthy, Powdery, Rust

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load Data
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Save Class Indices
    class_to_idx = train_dataset.class_to_idx
    with open(class_indices_path, 'w') as f:
        json.dump(class_to_idx, f)
    print(f"Class indices saved to {class_indices_path}: {class_to_idx}")

    # 2. CNN Model Architecture (Transfer Learning with ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training Loop
    print(f"Starting Training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")

    # 4. Save Model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train_model()
