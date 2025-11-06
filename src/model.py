import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvStem(nn.Module):
    """A CNN feature extractor used as the front-end before the Vision Transformer."""
    def __init__(self, in_ch=3, base_ch=64, out_ch=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class ViTEncoder(nn.Module):
    """Lightweight Vision Transformer Encoder."""
    def __init__(self, in_channels, feat_hw=(56, 56), patch_size=2, d_model=384, depth=4, n_heads=6):
        super().__init__()
        H, W = feat_hw
        self.patch_h = H // patch_size
        self.patch_w = W // patch_size
        self.num_patches = self.patch_h * self.patch_w
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_size = H // self.patch_h
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        patches = patches.transpose(1, 2)
        tokens = self.proj(patches)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.pos_embed
        out = self.encoder(tokens)
        return self.norm(out[:, 0])

class HybridCNNViT(nn.Module):
    """Combines CNN stem and Vision Transformer encoder for binary classification."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = ConvStem()
        self.vit = ViTEncoder(in_channels=256)
        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.vit(x)
        return self.head(x)
