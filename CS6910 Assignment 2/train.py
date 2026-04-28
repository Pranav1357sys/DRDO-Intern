import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import wandb

from models.cnn_model import CNNModel
from utils.data_loader import get_data_loaders

# ------------------ FIX RANDOMNESS ------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------ INIT WANDB ------------------
wandb.init(
    project="cs6910_assignment2_partA",
    config={
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "model": "CNN"
    }
)

config = wandb.config

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ DATA ------------------
train_loader, val_loader = get_data_loaders(
    "data/nature_12K/inaturalist_12K",
    config.batch_size
)

# ------------------ MODEL ------------------
model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

best_val_acc = 0

# ------------------ TRAIN ------------------
for epoch in range(config.epochs):

    model.train()
    correct_train = 0
    total_train = 0
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train

    # ------------------ VALIDATION ------------------
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_acc = correct_val / total_val

    # ------------------ LOG TO WANDB ------------------
    wandb.log({
        "epoch": epoch + 1,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "loss": running_loss
    })

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

# ------------------ TEST ------------------
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

correct_test = 0
total_test = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct_test += (preds == labels).sum().item()
        total_test += labels.size(0)

test_acc = correct_test / total_test

wandb.log({"test_acc": test_acc})
print(f"Test Accuracy: {test_acc:.4f}")

wandb.finish()