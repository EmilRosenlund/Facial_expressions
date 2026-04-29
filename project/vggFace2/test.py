import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for macOS OpenMP issues
import torch
import numpy as np
from train import VGGFace2WithMLP
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import random
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchvision import transforms
from dataloader import FER2013Dataset
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Model and data parameters
hidden_size = 128  
num_classes = 7
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
model_path = "best_model_v5.pth"
model_path = os.path.join(root_dir, model_path)
data_test_path = "small_test_set"
sample_count = 10
samples_root_dir = os.path.join(root_dir, "random_test_samples")


# Data transforms 
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load test images and labels
def load_test_images():
    dataset = FER2013Dataset(debug=False)
    default_data_path = os.path.join(root_dir, data_test_path)
    if os.path.exists(default_data_path):
        dataset.data_path = default_data_path
    originals = []
    images = []
    labels = []
    for expression, label in zip(class_names, range(num_classes)):
        data = dataset.load_data(split="test", expression=expression)
        for lbl, img in data:
            originals.append(img.convert('RGB').copy())
            images.append(transform(img.convert('RGB')))
            labels.append(label)
    return originals, images, labels
class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

orig_images, images, labels = load_test_images()
if len(images) == 0:
    raise RuntimeError(
        f"No test images found. Checked FER2013 path: {os.path.join(root_dir, 'fer2013')}"
    )

X_test_tensor = torch.stack(images)
y_test_tensor = torch.tensor(labels, dtype=torch.long)

# Load model
model = VGGFace2WithMLP(hidden_size, num_classes)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

# Predict in batches

batch_size = 64
preds = []
top_probs = []
with torch.no_grad():
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i+batch_size]
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)
        preds.append(torch.argmax(outputs, dim=1).cpu())
        top_probs.append(torch.max(probs, dim=1).values.cpu())
preds = torch.cat(preds).numpy()
top_probs = torch.cat(top_probs).numpy()
y_true = y_test_tensor.cpu().numpy()

print("Predicted classes:", np.unique(preds))

# Metrics
acc = accuracy_score(y_true, preds)
recall = recall_score(y_true, preds, average='macro')
precision = precision_score(y_true, preds, average='macro')
f1 = f1_score(y_true, preds, average='macro')
report = classification_report(y_true, preds, target_names=class_names)
cm = confusion_matrix(y_true, preds)

date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
filename = f"test_results_{date_time}.txt"
confusion_image_filename = f"confusion_matrix_{date_time}.png"
samples_dir = os.path.join(samples_root_dir, f"samples_{date_time}")
os.makedirs(samples_dir, exist_ok=True)
print(f"Test results at {date_time.replace('_', ' ')}:")

# Save 20 random test samples with predicted class and probability on the image.
sample_total = min(sample_count, len(orig_images))
sample_indices = random.sample(range(len(orig_images)), k=sample_total)
for i, idx in enumerate(sample_indices, start=1):
    img = orig_images[idx].copy()
    pred_label = class_names[preds[idx]]
    pred_prob = float(top_probs[idx]) * 100.0
    true_label = class_names[y_true[idx]]
    text_lines = [
        f"Pred: {pred_label} ({pred_prob:.2f}%)",
        f"True: {true_label}",
    ]

    font = ImageFont.load_default()
    padding = 8

    #footer below the image.
    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_heights = []
    max_text_w = 0
    for line in text_lines:
        left, top, right, bottom = measure_draw.textbbox((0, 0), line, font=font)
        max_text_w = max(max_text_w, right - left)
        line_heights.append(bottom - top)

    footer_h = sum(line_heights) + (len(text_lines) - 1) * 4 + 2 * padding
    canvas_w = max(img.width, max_text_w + 2 * padding)
    canvas_h = img.height + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    canvas.paste(img, ((canvas_w - img.width) // 2, 0))

    draw = ImageDraw.Draw(canvas)
    y_cursor = img.height + padding
    for line, h in zip(text_lines, line_heights):
        draw.text((padding, y_cursor), line, fill=(255, 255, 255), font=font)
        y_cursor += h + 4

    sample_name = f"sample_{i:02d}_{pred_label}_{pred_prob:.2f}.jpg"
    canvas.save(os.path.join(samples_dir, sample_name))

# Save confusion matrix
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(confusion_image_filename, dpi=300)
plt.close(fig)

# Save results
with open(filename, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Recall (macro): {recall:.4f}\n")
    f.write(f"Precision (macro): {precision:.4f}\n")
    f.write(f"F1 (macro): {f1:.4f}\n\n")
    f.write(report)
    f.write(f"\nRandom sample outputs: {samples_dir}\n")

print(f"Test results saved to {filename}")
print(f"Confusion matrix image saved to {confusion_image_filename}")
print(f"Saved {sample_total} annotated random samples to {samples_dir}")


