import random


import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
import sys
import os
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import FER2013Dataset
from torch.utils.data import Dataset, DataLoader

def mixup_batch(inputs, labels, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0)).to(inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[idx]
    return mixed, labels, labels[idx], lam

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


# Combined model: VGGFace2 (InceptionResnetV1) backbone + MLP head
class VGGFace2WithMLP(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_rate=0.4):
        super().__init__()
        backbone = InceptionResnetV1(pretrained='vggface2')
        # Unfreeze last blocks: last 2 mixed_6 and all mixed_7 layers
        for name, param in backbone.named_parameters():
            param.requires_grad = False
        for name, param in backbone.named_parameters():
            if any([k in name for k in ["block8", "mixed_7a", "mixed_6a", "conv2d_4b", "conv2d_4a"]]):
                param.requires_grad = True
        self.dropout = nn.Dropout(dropout_rate)
        self.backbone = backbone
        self.head = MLPHead(512, hidden_size, num_classes, dropout_rate)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.head(x)
        return x

# Custom dataset for end-to-end training
class FER2013ImageDataset(Dataset):
    def __init__(self, split="train", transform=None, augment=False):
        self.dataset = FER2013Dataset(debug=False)
        self.transform = transform
        self.augment = augment
        self.samples = []
        for expression in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            data = self.dataset.load_data(split=split, expression=expression)
            for label, img in data:
                self.samples.append((img.copy(), label))
                # For training, add 3 augmentations per image
                if self.augment and split == "train":
                    for aug_img in self._augmentations(img.copy()):
                        self.samples.append((aug_img, label))

    def _augmentations(self, img):
        # Squeeze (random horizontal scaling)
        squeeze_factor = random.uniform(0.7, 1.0)
        squeeze_img = img.resize((int(img.width * squeeze_factor), img.height)).resize((img.width, img.height))
        # Rotate
        rotate_img = img.rotate(random.uniform(-20, 20))
        # Add Gaussian noise
        np_img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 10, np_img.shape)
        noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img)
        return [squeeze_img, rotate_img, noisy_img]

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
            # MixUp augmentation
            mixed_inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels)
            outputs = model(mixed_inputs)
            # No log_softmax here, CrossEntropyLoss expects logits
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
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
            print(f"New best model found at epoch {epoch+1} with val loss {avg_val_loss:.4f}. Saving model...v5")
            torch.save(model.state_dict(), "best_model_v5.pth")

    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    hidden_size = 128
    num_classes = 7
    dropout_rate = 0.5
    random_seed = 42
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    print("Data transforms defined.")
    # Datasets and loaders with 3x augmentation for training
    full_train_dataset = FER2013ImageDataset(split="train", transform=transform, augment=True)
    val_split = 0.2
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
    # Model
    model = VGGFace2WithMLP(hidden_size, num_classes, dropout_rate)
    print(f"Model initialized. With VGGFace2 backbone and MLP head. and {sum(p.numel() for p in model.parameters())} parameters.")
    class_counts = torch.tensor([3995, 436, 4097, 7215, 4965, 4830, 3171], dtype=torch.float)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights.to(device))

        # Early backbone (frozen): lr=0, mid backbone: lr=1e-5, last block: lr=1e-4
    last_block_names = ["block8", "mixed_7a"]
    mid_block_names = ["mixed_6a", "conv2d_4b", "conv2d_4a"]
    # Group parameters
    last_block_params = [p for n, p in model.backbone.named_parameters() if any(k in n for k in last_block_names)]
    mid_block_params = [p for n, p in model.backbone.named_parameters() if any(k in n for k in mid_block_names) and not any(k in n for k in last_block_names)]
    frozen_params = [p for n, p in model.backbone.named_parameters() if not any(k in n for k in (last_block_names + mid_block_names))]
    print("[DEBUG] Last block params unfrozen:")
    for n, p in model.backbone.named_parameters():
        if any(k in n for k in last_block_names):
            print("  ", n)
    print("[DEBUG] Mid block params unfrozen:")
    for n, p in model.backbone.named_parameters():
        if any(k in n for k in mid_block_names) and not any(k in n for k in last_block_names):
            print("  ", n)
    print("[DEBUG] Frozen block params:")
    for n, p in model.backbone.named_parameters():
        if any(k in n for k in frozen_params):
            print("  ", n)

    backbone_param_groups = [
        {"params": frozen_params, "lr": 0},
        {"params": mid_block_params, "lr": 1e-5},
        {"params": last_block_params, "lr": 1e-4},
    ]
    optimizer = optim.AdamW(
        backbone_param_groups + [
            {"params": model.head.parameters(), "lr": 1e-4}
        ],
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_epochs = 300

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)