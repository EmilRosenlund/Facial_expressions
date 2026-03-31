import torch
import numpy as np
from train import VGGFace2WithMLP
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import os
from torchvision import transforms
from dataloader import FER2013Dataset
from PIL import Image

# Model and data parameters
hidden_size = 128  # Should match training
num_classes = 7
model_path = "best_model_v4.pth"

# Data transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load test images and labels
def load_test_images():
    dataset = FER2013Dataset(debug=False)
    images = []
    labels = []
    for expression, label in zip(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"], range(num_classes)):
        data = dataset.load_data(split="test", expression=expression)
        for lbl, img in data:
            images.append(transform(img.convert('RGB')))
            labels.append(label)
    return images, labels

images, labels = load_test_images()
X_test_tensor = torch.stack(images)
y_test_tensor = torch.tensor(labels, dtype=torch.long)

# Load model
model = VGGFace2WithMLP(hidden_size, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Predict in batches (avoid OOM)

batch_size = 64
preds = []
with torch.no_grad():
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i+batch_size]
        outputs = model(batch)
        preds.append(torch.argmax(outputs, dim=1).cpu())
preds = torch.cat(preds).numpy()
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
