# src/siamese_diff/model.py

# Third-party
import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(nn.Module):
    """Siamese Neural Network for image similarity."""

    def __init__(self) -> None:
        super().__init__()
        # Pretrained ResNet-18 backbone
        backbone: models.ResNet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        backbone.fc = nn.Linear(512, 128)  # Embed to 128-dim vector
        self.backbone: nn.Module = backbone

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single image."""
        return self.backbone(x)

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a pair of images."""
        out1: torch.Tensor = self.forward_once(img1)
        out2: torch.Tensor = self.forward_once(img2)
        return out1, out2
