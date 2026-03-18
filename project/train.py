import torch
import torch.nn as nn
import numpy as np
from dataloader import FER2013Dataset
from sklearn.model_selection import train_test_split

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
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
                val_outputs = model(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        model.train()

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")

    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    input_size = 512  # Example input size (e.g., from a feature extractor)
    hidden_size = 256
    num_classes = 7  # Number of emotion classes

    model = SimpleMLP(input_size, hidden_size, num_classes)

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
            y.append(np.full(embeddings.shape[0], label))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Split into train and validation sets (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs)