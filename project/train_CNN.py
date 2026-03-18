import torch
import torch.nn as nn
import numpy as np
import time
from dataloader import FER2013Dataset
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------ #
# Model                                                                #
# ------------------------------------------------------------------ #

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        # Input: (B, 512) → reshaped to (B, 1, 32, 16)
        self.input_channels = 1
        self.height         = 32
        self.width          = 16

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_rate),
        )

        self.pool    = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1     = nn.Linear(128 * 2 * 2, 64)
        self.bn_fc   = nn.BatchNorm1d(64)
        self.act     = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.fc2(x)


# ------------------------------------------------------------------ #
# Label smoothing                                                      #
# ------------------------------------------------------------------ #

def smooth_labels(
    labels:     torch.Tensor,
    num_classes: int,
    smoothing:  float = 0.1,
) -> torch.Tensor:
    with torch.no_grad():
        confidence = 1.0 - smoothing
        smoothed   = torch.full(
            (labels.size(0), num_classes),
            smoothing / (num_classes - 1),
        )
        smoothed.scatter_(1, labels.unsqueeze(1), confidence)
    return smoothed


# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #

EXPRESSIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_data(
    device:      torch.device,
    num_classes: int = 7,
    test_size:   float = 0.2,
    random_seed: int = 42,
) -> tuple:
    """
    Load embeddings from disk, convert to float32, split into
    train/val, move to device, and free all intermediate arrays.

    Returns:
        X_train_tensor, y_train_tensor,
        X_val_tensor,   y_val_tensor,
        class_weights_row  (1, num_classes) on device
    """
    dataset      = FER2013Dataset()
    X_list       = []
    y_list       = []
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for idx, exp in enumerate(EXPRESSIONS):
        emb = dataset.load_embeddings(split="train", expression=exp)
        if emb is None or emb.size == 0:
            print(f"  Warning: no embeddings found for '{exp}' — skipping")
            continue

        # Squeeze extra dim if present
        if emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)

        # Convert to float32 immediately — halves memory vs float64
        # copy=False avoids an extra copy if already float32
        emb = emb.astype(np.float32, copy=False)

        class_counts[idx] = len(emb)
        X_list.append(emb)

        labels_tensor = torch.full((len(emb),), idx, dtype=torch.long)
        smoothed      = smooth_labels(labels_tensor, num_classes)
        y_list.append(smoothed.numpy().astype(np.float32, copy=False))

    # Concatenate then immediately free the per-class lists
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    del X_list, y_list

    print(f"  Total samples : {len(X)}")
    print(f"  Class counts  : {class_counts}")
    print(f"  X dtype       : {X.dtype}  shape: {X.shape}")

    # Split — stratify on hard labels for balanced splits
    hard_labels = np.argmax(y, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_seed,
        stratify     = hard_labels,
    )
    # Free full arrays immediately
    del X, y, hard_labels

    # torch.from_numpy shares memory (no copy) since already float32
    # .to(device) makes one copy to device — then we free the numpy arrays
    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    X_val_tensor   = torch.from_numpy(X_val).to(device)
    y_val_tensor   = torch.from_numpy(y_val).to(device)
    del X_train, X_val, y_train, y_val

    # Class weights — computed from counts recorded during loading
    weights = 1.0 / (class_counts + 1e-6)
    weights = (weights / weights.sum() * num_classes).astype(np.float32)
    class_weights_row = (
        torch.from_numpy(weights)
        .to(device)
        .unsqueeze(0)   # (1, num_classes) — ready for broadcasting in loss
    )

    print(f"  Train samples : {len(X_train_tensor)}")
    print(f"  Val samples   : {len(X_val_tensor)}")
    print(f"  Class weights : {weights.round(3)}")

    return (
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        class_weights_row,
    )


# ------------------------------------------------------------------ #
# Training                                                             #
# ------------------------------------------------------------------ #

def train(
    model:             nn.Module,
    train_loader:      torch.utils.data.DataLoader,
    val_loader:        torch.utils.data.DataLoader,
    criterion:         nn.Module,
    optimizer:         torch.optim.Optimizer,
    scheduler,
    num_epochs:        int,
    class_weights_row: torch.Tensor,
    device:            torch.device,
) -> None:
    """
    class_weights_row : (1, num_classes) tensor already on device
    """
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ---------------------------------------------------------- #
        # Training                                                     #
        # ---------------------------------------------------------- #
        model.train()
        epoch_start = time.time()
        total_loss  = torch.tensor(0.0, device=device)

        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)

            outputs         = torch.log_softmax(model(inputs), dim=1)
            weighted_labels = labels * class_weights_row
            loss            = criterion(outputs, weighted_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Stay on device — no .item() inside the loop
            total_loss += loss.detach()

        # Sync before reading timing
        if device.type == "mps":
            torch.mps.synchronize()

        avg_train_loss = (total_loss / len(train_loader)).item()

        # ---------------------------------------------------------- #
        # Validation                                                   #
        # ---------------------------------------------------------- #
        model.eval()
        val_loss = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs     = torch.log_softmax(model(val_inputs), dim=1)
                weighted_labels = val_labels * class_weights_row
                loss            = criterion(val_outputs, weighted_labels)
                val_loss       += loss.detach()

        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        avg_val_loss = (val_loss / len(val_loader)).item()
        epoch_time   = time.time() - epoch_start

        print(
            f"Epoch [{epoch+1:>3}/{num_epochs}] "
            f"train={avg_train_loss:.4f}  "
            f"val={avg_val_loss:.4f}  "
            f"time={epoch_time:.2f}s"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model_cnn.pth")
            print(f"  ✓ saved best model (val={avg_val_loss:.4f})")

    torch.save(model.state_dict(), "model.pth")
    print("Training complete.")


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    NUM_CLASSES  = 7
    DROPOUT_RATE = 0.5
    BATCH_SIZE   = 4096
    NUM_EPOCHS   = 200
    LR           = 1e-3
    WEIGHT_DECAY = 5e-4
    RANDOM_SEED  = 42

    torch.manual_seed(RANDOM_SEED)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load data — all intermediates freed inside load_data()
    print("\nLoading embeddings...")
    (
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        class_weights_row,
    ) = load_data(device=device, num_classes=NUM_CLASSES, random_seed=RANDOM_SEED)

    # DataLoaders — data already on device so num_workers=0
    batch_size   = min(len(X_train_tensor), BATCH_SIZE)
    train_ds     = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_ds       = torch.utils.data.TensorDataset(X_val_tensor,   y_val_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = SimpleCNN(NUM_CLASSES, DROPOUT_RATE).to(device)

    # Training components
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    print("\nStarting training...\n")
    train(
        model             = model,
        train_loader      = train_loader,
        val_loader        = val_loader,
        criterion         = criterion,
        optimizer         = optimizer,
        scheduler         = scheduler,
        num_epochs        = NUM_EPOCHS,
        class_weights_row = class_weights_row,
        device            = device,
    )