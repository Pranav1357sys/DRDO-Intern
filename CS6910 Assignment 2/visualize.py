import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from models.cnn_model import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data_dir = "data/nature_12K/inaturalist_12K"

test_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

class_names = test_dataset.classes

model = CNNModel(num_classes=10).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

images = images.cpu()

# Plot 10x3 grid
fig, axes = plt.subplots(10, 3, figsize=(10, 20))

for i, ax in enumerate(axes.flat):
    img = images[i].permute(1, 2, 0)
    ax.imshow(img)
    ax.set_title(f"P: {class_names[preds[i]]}")
    ax.axis("off")

plt.tight_layout()
plt.show()