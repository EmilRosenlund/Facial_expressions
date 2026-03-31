import matplotlib.pyplot as plt

# Read training and validation loss from training_log.txt
train_losses = []
val_losses = []
epochs = []

with open("training_log.txt", "r") as f:
    for line in f:
        if "Epoch" in line and "Train Loss" in line and "Val Loss" in line:
            # Example line: Epoch [1/50], Train Loss: 1.2345, Val Loss: 1.5678
            parts = line.strip().split(',')
            epoch_part = parts[0].split('[')[-1].split('/')[0]
            train_loss = float(parts[1].split(':')[-1])
            val_loss = float(parts[2].split(':')[-1])
            epochs.append(int(epoch_part))
            train_losses.append(train_loss)
            val_losses.append(val_loss)

plt.figure(figsize=(10,6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
