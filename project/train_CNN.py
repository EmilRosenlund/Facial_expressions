import torch
import torch.nn as nn
import numpy as np
from dataloader import FER2013Dataset
from sklearn.model_selection import train_test_split

class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.4):
        super(SimpleCNN, self).__init__()
        # Assume input_size=512, reshape to (1, 32, 16) for CNN (since 32*16=512)
        self.input_channels = 1
        self.height = 32
        self.width = 16
        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.act = nn.LeakyReLU()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(128 * 2 * 2, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch, 512]
        x = x.view(-1, self.input_channels, self.height, self.width)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.dropout(x)
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

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, class_weights_tensor):
    best_val_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.log_softmax(outputs, dim=1)
            # Apply class weights to the targets
            weighted_labels = labels * class_weights_tensor.to(labels.device).unsqueeze(0)  # [batch, num_classes]
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
                val_inputs = val_inputs.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)
                val_outputs = model(val_inputs)
                val_outputs = torch.log_softmax(val_outputs, dim=1)
                weighted_val_labels = val_labels * class_weights_tensor.to(val_labels.device).unsqueeze(0)
                loss = criterion(val_outputs, weighted_val_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)
        model.train()

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model found at epoch {epoch+1} with val loss {avg_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), "best_model_cnn.pth")

    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    input_size = 512  # Example input size (e.g., from a feature extractor)
    num_classes = 7  # Number of emotion classes
    dropout_rate = 0.5
    random_seed = 42
    torch.manual_seed(random_seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')  # For Apple Silicon (Mac M1/M2)
    else:
        device = torch.device('cpu')

    model = SimpleCNN(input_size, num_classes, dropout_rate)
    model.to(device)

    # Load embeddings and labels
    dataset = FER2013Dataset()
    X = []
    y = []
    for expression, label in zip(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"], range(num_classes)):
        embeddings = dataset.load_embeddings(split="train", expression=expression)
        if embeddings is not None:
            # Squeeze extra dimension if present
            if embeddings.ndim == 3 and embeddings.shape[1] == 1:
                embeddings = embeddings.squeeze(1)
            X.append(embeddings)
            # Create smoothed labels for the whole batch
            labels_tensor = torch.full((embeddings.shape[0],), label, dtype=torch.long)
            smoothed = smooth_labels(labels_tensor, num_classes, smoothing=0.1)
            y.append(smoothed.cpu().numpy())
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Split into train and validation sets (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Ensure float32 for KLDivLoss
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)      # Ensure float32 for KLDivLoss

    # Move tensors to device
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)

    # Check shapes for debugging
    print('X_train_tensor shape:', X_train_tensor.shape)
    print('y_train_tensor shape:', y_train_tensor.shape)
    print('X_val_tensor shape:', X_val_tensor.shape)
    print('y_val_tensor shape:', y_val_tensor.shape)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False)

    # Calculate class weights for imbalance
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for expression, label in zip(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"], range(num_classes)):
        embeddings = dataset.load_embeddings(split="train", expression=expression)
        if embeddings is not None:
            if embeddings.ndim == 3 and embeddings.shape[1] == 1:
                embeddings = embeddings.squeeze(1)
            class_counts[label] = embeddings.shape[0]
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print('Class counts:', class_counts)
    print('Class weights:', class_weights)

    # Training setup
    criterion = nn.KLDivLoss(reduction='batchmean')  # Use batchmean for KLDivLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4) # AdamW and stronger weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    num_epochs = 200

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, class_weights_tensor)