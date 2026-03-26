import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from CNN_classifier import ExpressionClassifier
from dataloader import FER2013Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import os
import time

class FERDatasetTorch(Dataset):
    def __init__(self, root_dir, split="test", transform=None):
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
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = FERDatasetTorch("fer2013", split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    model = ExpressionClassifier().to(device)
    model_path = "best_expression_cnn.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    print("Predicted classes:", np.unique(y_pred))

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])

    date_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    filename = f"test_CNN_results_{date_time}.txt"
    print(f"Test results at {date_time.replace('_', ' ')}:")
    with open(filename, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Recall (macro): {recall:.4f}\n")
        f.write(f"Precision (macro): {precision:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n\n")
        f.write(report)
    print(f"Test results saved to {filename}")

if __name__ == "__main__":
    main()
