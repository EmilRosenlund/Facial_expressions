import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionClassifier(nn.Module):
    """
    CNN for facial expression classification on 48x48 grayscale images.
    Output: 7 classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    """
    def __init__(self):
        super(ExpressionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        """Returns predicted class index for input batch."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

