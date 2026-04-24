# Sangeeth Deleep Menon (NUID: 002524579)
# Raj Gupta (NUID: 002068701)
# CS5330 Final Project - Model Architectures
# Spring 2026
#
# Defines three architectures for multi-class RGB object recognition:
#   ObjectCNN      - 3-layer CNN adapted from Project 5, input 3 x IMG_SIZE x IMG_SIZE
#   ObjectViT      - Minimal Vision Transformer (patch size 8), same input/output
#   MobileNetV2    - Pretrained MobileNetV2 backbone with a custom classifier head
#                    (best real-world generalisation from small datasets)
#
# All three return log-softmax probabilities over num_classes.
# MobileNetV2 expects IMG_SIZE = 224; CNN and ViT use IMG_SIZE = 64.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMG_SIZE        = 64    # spatial size for CNN and ViT
IMG_SIZE_MOBILE = 224   # spatial size for MobileNetV2


# ---------------------------------------------------------------------------
# ObjectCNN
# ---------------------------------------------------------------------------

class ObjectCNN(nn.Module):
    """
    Three-layer CNN for multi-class RGB object recognition.

    Architecture (input 3 x 64 x 64):
        Conv(3->32, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)   : 64 -> 32
        Conv(32->64, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2)  : 32 -> 16
        Conv(64->128, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2) : 16 -> 8
        Flatten: 128 * 8 * 8 = 8192
        FC(8192 -> 256) -> ReLU -> Dropout(0.4)
        FC(256 -> num_classes) -> log_softmax
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.4)
        self.fc1   = nn.Linear(128 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # -> 8x8
        x = x.view(x.size(0), -1)                        # -> 8192
        x = self.drop(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)


# ---------------------------------------------------------------------------
# ObjectViT
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and linearly projects each."""

    def __init__(self, img_size=IMG_SIZE, patch_size=8, in_ch=3, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                          # (B, embed_dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)       # (B, n_patches, embed_dim)


class ObjectViT(nn.Module):
    """
    Minimal Vision Transformer for multi-class RGB object recognition.

    Architecture (input 3 x 64 x 64, patch_size=8):
        PatchEmbed: 64 patches of dim 128
        CLS token prepended -> sequence length 65
        Learned positional embeddings
        4 x TransformerEncoderLayer (d=128, heads=4, ff=512)
        LayerNorm -> CLS token -> Linear -> log_softmax
    """

    def __init__(self, num_classes, img_size=IMG_SIZE, patch_size=8,
                 embed_dim=128, depth=4, n_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = _PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x   = self.patch_embed(x)                         # (B, n_patches, D)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                  # (B, n_patches+1, D)
        x   = self.pos_drop(x + self.pos_embed)
        x   = self.transformer(x)
        x   = self.norm(x[:, 0])                          # CLS token output
        return F.log_softmax(self.head(x), dim=1)


# ---------------------------------------------------------------------------
# MobileNetV2 (transfer learning)
# ---------------------------------------------------------------------------

class MobileNetV2(nn.Module):
    """
    Pretrained MobileNetV2 backbone with a fine-tuned classification head.

    The ImageNet backbone provides strong general visual features.
    Only the final classifier layer is replaced; the backbone is fine-tuned
    with a low learning rate to adapt to the target classes.

    Input:  3 x 224 x 224 (use IMG_SIZE_MOBILE = 224)
    Output: num_classes log-probabilities
    """

    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in base.features.parameters():
                param.requires_grad = False

        # Replace the default 1000-class head with our own
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        self.model = base

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)
