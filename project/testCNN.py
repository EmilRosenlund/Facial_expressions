import torch
import numpy as np
from dataloader import FER2013Dataset
from train_CNN import SimpleCNN 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# Model and data parameters
input_size = 512
hidden_size = 256
num_classes = 7
model_path = "best_model_cnn.pth"

# Load test embeddings and labels

dataset = FER2013Dataset()
X_test = []
y_test = []
for expression, label in zip(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"], range(num_classes)):
    embeddings = dataset.load_embeddings(split="test", expression=expression)
    if embeddings is not None:
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)
        X_test.append(embeddings)
        y_test.append(np.full(embeddings.shape[0], label))
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

# Convert to torch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Load model
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Predict
with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

print("Predicted classes:", np.unique(preds))

# Metrics
acc = accuracy_score(y_true, preds)
recall = recall_score(y_true, preds, average='macro')
precision = precision_score(y_true, preds, average='macro')
f1 = f1_score(y_true, preds, average='macro')
report = classification_report(y_true, preds, target_names=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])
import time

date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
filename = f"test_results_{date_time}.txt"
print(f"Test results at {date_time.replace('_', ' ')}:")
# Save results
with open(filename, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Recall (macro): {recall:.4f}\n")
    f.write(f"Precision (macro): {precision:.4f}\n")
    f.write(f"F1 (macro): {f1:.4f}\n\n")
    f.write(report)

print(f"Test results saved to {filename}")
