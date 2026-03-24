import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from dataloader import FER2013Dataset
from CNN_classifier import ExpressionClassifier

class FERDatasetTorch(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.label_map = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "sad": 4, "surprise": 5, "neutral": 6}
        for label in self.label_map:
            dir_path = os.path.join(root_dir, split, label)
            if os.path.exists(dir_path):
                for fname in os.listdir(dir_path):
                    if fname.endswith(".jpg"):
                        self.samples.append((os.path.join(dir_path, fname), self.label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    import time
    day = time.strftime("%m-%d", time.localtime())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Time dataset loading
    t0 = time.time()
    full_dataset = FERDatasetTorch("fer2013", split="train", transform=transform)
    print(f"Loaded {len(full_dataset)} total training samples. (Took {time.time() - t0:.2f}s)")
    # Create train/val split (e.g., 80% train, 20% val)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # Try higher num_workers and larger batch size for speed
    batch_size = 128
    num_workers = min(8, os.cpu_count() or 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=="cuda"))
    model = ExpressionClassifier().to(device)
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters.")
    print("Layer-wise parameter counts:")
    for name, param in model.named_parameters():
        print(f"  {name:40s} {param.numel():,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4) # AdamW and stronger weight decay
    num_epochs = 100
    best_val_loss = float('inf')
    best_epoch = -1
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_loader.dataset)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_expression_cnn_{day}.pth")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        # Save final model as well (optional)
    torch.save(model.state_dict(), f"expression_cnn_{day}.pth")
    print("Training complete. Model saved as expression_cnn.pth")
    print(f"Best epoch: {best_epoch} with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
