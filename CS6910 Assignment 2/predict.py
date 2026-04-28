import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)

# Modify final layer (IMPORTANT - same as training)
num_classes = 10  # ⚠️ CHANGE THIS if your dataset has different classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load weights
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])

# Class names (AUTO LOAD)
train_path = "data/nature_12K/inaturalist_12K/train"
class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

# TEST
if __name__ == "__main__":
    img_path = input("Enter image path: ")
    result = predict(img_path)
    print("Predicted class:", result)