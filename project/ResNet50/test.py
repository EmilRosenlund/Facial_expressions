import torch
import numpy as np
from train import SimpleMLP
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import os

# Model and data parameters
input_size = 2048  # ResNet50 embedding size
hidden_size = 128
num_classes = 7
model_path = "best_model.pth"

# Load test embeddings and labels
def load_resnet50_embeddings(split="test"):
    X = []
    y = []
    for expression, label in zip(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"], range(num_classes)):
        embedding_path = f"embeddings_ResNet50/embeddings_{split}_{expression}.npy"
        if not os.path.exists(embedding_path):
            print(f"Warning: {embedding_path} not found, skipping.")
            continue
        embeddings = np.load(embedding_path)
        if embeddings.shape[1] != 2048:
            raise ValueError(f"Embeddings for {expression} have shape {embeddings.shape}, expected (N, 2048)")
        X.append(embeddings)
        y.append(np.full(embeddings.shape[0], label))
    if not X:
        raise RuntimeError("No test embeddings found!")
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

X_test, y_test = load_resnet50_embeddings(split="test")

# Convert to torch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Load model
model = SimpleMLP(input_size, hidden_size, num_classes)
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
