from tokenize import Double

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

data_dir = 'data/processed'
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'test')
val_path = os.path.join(data_dir, 'valid')

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val_test': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    'train': datasets.ImageFolder(train_path, transform=data_transforms['train']),
    'val': datasets.ImageFolder(val_path, transform=data_transforms['val_test']),
    'test': datasets.ImageFolder(test_path, transform=data_transforms['val_test'])
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val', 'test']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Start training op: {device}")
for epoch in range(3):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(image_datasets['train'])
    print(f'Epoch {epoch+1}/3, Loss: {epoch_loss:.4f}')

os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/parking_model.pth')
print("Model opgeslagen in models/parking_model.pth")
