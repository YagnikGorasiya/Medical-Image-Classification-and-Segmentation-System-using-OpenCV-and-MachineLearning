import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

dataset_path = "datasets/chest_xray"

train_ds = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
val_ds = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0

for epoch in range(3):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        os.makedirs("backend/models", exist_ok=True)
        torch.save(model.state_dict(), "backend/models/chest_cnn.pt")
        print("✅ Saved best model to backend/models/chest_cnn.pt")

print("Training Complete!")
print("Best Accuracy:", best_acc)
print("Classes:", train_ds.classes)
