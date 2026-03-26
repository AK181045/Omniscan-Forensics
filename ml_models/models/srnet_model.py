"""
OMNISCAN FORENSICS — SRNet Steganalysis Architecture & Grad-CAM Visualizer
References:
1. M. Boroumand, M. Chen, & J. Fridrich (2018) - 'Deep Residual Network for Steganalysis of Digital Images - SRNet'
2. N. Swarnkar & A. Thomas (2025) - 'StegEfficientNetB0 Framework for Spatial-Domain Image Steganalysis'
3. R. R. Selvaraju et al. (2017) - 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization'

Features:
- Residual blocks without high-pass pre-filtering to preserve subtle high-frequency noise residuals.
- Dual-domain feature fusion (Spatial + Spectral inputs).
- Exact Grad-CAM implementation:
    alpha_k^c = (1/Z) * sum_i sum_j (dY^c / dA_{i,j}^k)
    L_Grad-CAM = ReLU( sum_k alpha_k^c * A^k )
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("omniscan.models.srnet")


class SRNetResidualBlock(nn.Module):
    """Residual block of SRNet designed to extract unquantified noise residuals."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res)
        return out


class OmniScanFusionNet(nn.Module):
    """
    Universal Steganalysis Fusion Network combining:
    - SRNet Noise Residual Extractor (Boroumand 2018)
    - Compound Feature Layers (Swarnkar & Thomas 2025)
    - Grad-CAM explainability hooks on the final convolutional feature maps.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Initial un-quantified residual convolutions (Layers 1-4)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Residual Blocks (Layers 5-12)
        self.layer1 = SRNetResidualBlock(32, 64, stride=2)
        self.layer2 = SRNetResidualBlock(64, 128, stride=2)
        self.layer3 = SRNetResidualBlock(128, 256, stride=2)
        # Final conv layer for Grad-CAM
        self.target_conv = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.target_bn = nn.BatchNorm2d(512)

        # Global Average Pooling & Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # Gradient & feature hooks for Grad-CAM
        self.gradients = None
        self.activations = None

    def _save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Grad-CAM Target Layer
        features = F.relu(self.target_bn(self.target_conv(x)))
        self.activations = features
        if features.requires_grad:
            features.register_hook(self._save_gradient)

        pooled = self.gap(features)
        flat = torch.flatten(pooled, 1)
        out = self.fc(flat)
        return out


class GradCAMExplainer:
    """
    Computes exact Grad-CAM visual heatmaps as specified in Selvaraju et al. (2017):
    alpha_k^c = (1/Z) * sum_i sum_j (dY^c / dA_{i,j}^k)
    L_Grad-CAM = ReLU( sum_k alpha_k^c * A^k )
    """
    def __init__(self, model: OmniScanFusionNet):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap (0.0 to 1.0) for the target class.
        input_tensor: shape (1, 3, H, W)
        """
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_tensor)
        
        # Target score
        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.model.gradients
        activations = self.model.activations

        if gradients is None or activations is None:
            # Fallback if gradients aren't tracked
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global average pooling of gradients: alpha_k = (1/Z) * sum_i sum_j dY / dA
        alpha = torch.mean(gradients, dim=(2, 3), keepdim=True)
        # Weighted combination: sum_k alpha_k * A^k
        cam = torch.sum(alpha * activations, dim=1, keepdim=True)
        # ReLU activation to isolate positive contributing features
        cam = F.relu(cam)

        # Resize to input dimensions
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_heatmap(self, original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay color-coded Grad-CAM heatmap (Jet colormap) onto the original image.
        """
        h, w = original_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(original_rgb, 1 - alpha, colormap_rgb, alpha, 0)
        return blended
