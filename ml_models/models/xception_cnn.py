"""
SENTINELS OF INTEGRITY — Xception-based CNN Deepfake Detector
Backbone: Xception (Design.txt §3: "Xception-based CNN")
Input: 299x299 pixel frames
"""

import torch
import torch.nn as nn
# import torchvision.models as models  # Removed for compatibility


class XceptionBlock(nn.Module):
    """Depthwise separable convolution block used in Xception."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out + residual)
        return out


class XceptionCNN(nn.Module):
    """
    Xception-based CNN for spatial deepfake detection.

    Analyzes individual frames for:
    - Face inconsistencies
    - Edge blending artifacts
    - Ghosting patterns
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Middle flow (Xception blocks)
        self.middle = nn.Sequential(
            XceptionBlock(64, 128, stride=2),
            XceptionBlock(128, 256, stride=2),
            XceptionBlock(256, 512, stride=2),
            XceptionBlock(512, 728, stride=1),
            XceptionBlock(728, 728, stride=1),
            XceptionBlock(728, 728, stride=1),
            XceptionBlock(728, 728, stride=1),
        )

        # Exit flow
        self.exit = nn.Sequential(
            XceptionBlock(728, 1024, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Feature extractor output (for ensemble)
        self.feature_dim = 1024

    def extract_features(self, x):
        """Extract spatial features without classification."""
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        return x.flatten(1)  # (batch, 1024)

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def get_spatial_scores(self, x):
        """Get per-frame spatial analysis scores."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            return {
                "is_synthetic": bool(probs[0, 1] > 0.5),
                "confidence": float(probs[0, 1]),
                "real_prob": float(probs[0, 0]),
                "fake_prob": float(probs[0, 1]),
            }
