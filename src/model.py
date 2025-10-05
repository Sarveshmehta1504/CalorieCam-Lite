import torch
import torch.nn as nn
from torchvision.models import resnet18

class EmbeddingNet(nn.Module):
    def __init__(self, pretrained=True, embed_dim=512):
        super().__init__()
        m = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # Remove final fc, keep 512-dim features
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.embed_dim = 512

    def forward(self, x):
        # Bx512x1x1 -> Bx512
        feats = self.backbone(x).flatten(1)
        return feats

class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, emb):
        return self.fc(emb)

class EmbeddingClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embed = EmbeddingNet(pretrained=True)
        self.head = LinearHead(self.embed.embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)
        logits = self.head(emb)
        return logits, emb
