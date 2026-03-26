
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import numpy as np
import sys
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import FER2013Dataset
from torch.utils.data import Dataset, DataLoader

# MLP head for classification
class MLPHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn4 = nn.BatchNorm1d(hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# Combined model: ResNet50 backbone + MLP head
class ResNet50WithMLP(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_rate=0.4):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Unfreeze last two residual blocks (layer3, layer4)
        for name, param in resnet.named_parameters():
            param.requires_grad = False
        for name, param in resnet.layer3.named_parameters():
            param.requires_grad = True
        for name, param in resnet.layer4.named_parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout_rate)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))  # Remove FC
        self.head = MLPHead(2048, hidden_size, num_classes, dropout_rate)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

# Custom dataset for end-to-end training
class FER2013ImageDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.dataset = FER2013Dataset(debug=False)
        self.transform = transform
        self.samples = []
        for expression in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            data = self.dataset.load_data(split=split, expression=expression)
            for label, img in data:
                self.samples.append((img.copy(), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img.convert('RGB'))
        return img, label

def smooth_labels(labels, num_classes, smoothing=0.1):
    with torch.no_grad():
        confidence = 1.0 - smoothing
        label_shape = (labels.size(0), num_classes)
        smoothed = torch.full(label_shape, smoothing / (num_classes - 1), device=labels.device)
        smoothed.scatter_(1, labels.unsqueeze(1), confidence)
    return smoothed


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.log_softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_outputs = torch.log_softmax(val_outputs, dim=1)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model found at epoch {epoch+1} with val loss {avg_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), "best_model.pth")

    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    hidden_size = 128
    num_classes = 7
    dropout_rate = 0.5
    random_seed = 42
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Datasets and loaders
    full_train_dataset = FER2013ImageDataset(split="train", transform=transform)
    val_split = 0.2
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Model
    model = ResNet50WithMLP(hidden_size, num_classes, dropout_rate)

    # Loss (use label smoothing)
    criterion = nn.CrossEntropyLoss()

    # Optimizer: two parameter groups
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 1e-4}
    ], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_epochs = 50

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)