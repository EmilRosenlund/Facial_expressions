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
        # Encoder
        x1 = F.relu(self.conv1(x))  # (B, 64, 48, 48)
        p1 = self.pool(x1)          # (B, 64, 24, 24)
        x2 = F.relu(self.conv2(p1)) # (B, 128, 24, 24)
        p2 = self.pool(x2)          # (B, 128, 12, 12)
        p2 = self.dropout(p2)
        x3 = F.relu(self.conv3(p2)) # (B, 256, 12, 12)
        p3 = self.pool(x3)          # (B, 256, 6, 6)
        x4 = F.relu(self.conv4(p3)) # (B, 512, 6, 6)
        p4 = self.pool(x4)          # (B, 512, 3, 3)
        p4 = self.dropout(p4)

        # U-Net style skip connections (upsample and concatenate)
        up3 = F.interpolate(p4, size=x4.shape[2:], mode='bilinear', align_corners=False)  # (B, 512, 6, 6)
        cat3 = torch.cat([up3, x4], dim=1)  # (B, 1024, 6, 6)
        up2 = F.interpolate(cat3, size=x3.shape[2:], mode='bilinear', align_corners=False)  # (B, 1024, 12, 12)
        cat2 = torch.cat([up2, x3], dim=1)  # (B, 1280, 12, 12)
        up1 = F.interpolate(cat2, size=x2.shape[2:], mode='bilinear', align_corners=False)  # (B, 1280, 24, 24)
        cat1 = torch.cat([up1, x2], dim=1)  # (B, 1408, 24, 24)
        up0 = F.interpolate(cat1, size=x1.shape[2:], mode='bilinear', align_corners=False)  # (B, 1408, 48, 48)
        cat0 = torch.cat([up0, x1], dim=1)  # (B, 1472, 48, 48)

        # Global average pooling to reduce to (B, 1472, 3, 3)
        gap = F.adaptive_avg_pool2d(cat0, (3, 3))  # (B, 1472, 3, 3)
        flat = torch.flatten(gap, 1)
        x = F.relu(self.fc1(flat))
        x = self.fc2(x)
        return x

    def predict(self, x):
        """Returns predicted class index for input batch."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

