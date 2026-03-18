import torch
import torch.nn as nn
import numpy as np
import time
from dataloader import FER2013Dataset
from sklearn.model_selection import train_test_split

class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.4):
        super(SimpleCNN, self).__init__()
        self.input_channels = 1
        self.height = 32
        self.width = 16
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(128 * 2 * 2, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_channels, self.height, self.width)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def smooth_labels(labels, num_classes, smoothing=0.1):
    with torch.no_grad():
        confidence = 1.0 - smoothing
        label_shape = (labels.size(0), num_classes)
        smoothed = torch.full(label_shape, smoothing / (num_classes - 1), device=labels.device)
        smoothed.scatter_(1, labels.unsqueeze(1), confidence)
    return smoothed

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, class_weights_tensor, device):
    best_val_loss = float('inf')
    # Ensure weights are on the correct device once
    class_weights_tensor = class_weights_tensor.to(device).unsqueeze(0)

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        
        for inputs, labels in train_loader:
            # non_blocking=True only helps if tensors are pinned; 
            # if they are already on GPU, this is nearly instant.
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True) # Slightly faster than zero_grad()
            outputs = torch.log_softmax(model(inputs), dim=1)
            
            # Apply weights without moving tensor to device every batch
            weighted_labels = labels * class_weights_tensor
            
            loss = criterion(outputs, weighted_labels)
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
                val_outputs = torch.log_softmax(model(val_inputs), dim=1)
                weighted_val_labels = val_labels * class_weights_tensor
                loss = criterion(val_outputs, weighted_val_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_cnn.pth")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    input_size = 512
    num_classes = 7
    dropout_rate = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = SimpleCNN(input_size, num_classes, dropout_rate).to(device)

    dataset = FER2013Dataset()
    X, y = [], []
    
    expressions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    for idx, exp in enumerate(expressions):
        embeddings = dataset.load_embeddings(split="train", expression=exp)
        if embeddings is not None and embeddings.size > 0:
            if embeddings.ndim == 3 and embeddings.shape[1] == 1:
                embeddings = embeddings.squeeze(1)
            X.append(embeddings)
            
            labels_tensor = torch.full((embeddings.shape[0],), idx, dtype=torch.long)
            smoothed = smooth_labels(labels_tensor, num_classes)
            y.append(smoothed.cpu().numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Note: We keep these as CPU tensors for the DataLoader to handle
    # UNLESS the entire dataset fits easily in VRAM. 
    # Since they are pre-extracted embeddings, they likely fit.
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).float().to(device)

    # PERFORMANCE TIP: If data is already on GPU, num_workers MUST be 0.
    batch_size = min(len(X_train), 4096)
    
    train_ds = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_ds = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights calculation
    class_counts = np.array([len(X[i]) for i in range(len(X))]) # Simplified for logic
    # (Re-using your existing logic for class_counts is fine here)
    class_weights = 1.0 / (np.histogram(np.argmax(y, axis=1), bins=num_classes)[0] + 1e-6)
    class_weights = torch.tensor(class_weights / class_weights.sum() * num_classes, dtype=torch.float32)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, 200, class_weights, device)