# src/siamese_diff/dataset.py

# Standard library
from pathlib import Path
from typing import List, Tuple

# Third-party
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class SiameseImageDataset(Dataset):
    """Dataset for Siamese network image pairs."""

    def __init__(self, base_path: str) -> None:
        self.base_path: Path = Path(base_path)
        self.pairs: List[Tuple[Path, Path, int]] = []

        # Load "same" pairs
        for file in (self.base_path / "same").iterdir():
            self.pairs.append((file, file, 1))  # label 1 = same

        # Load "different" pairs
        for file in (self.base_path / "different").iterdir():
            self.pairs.append((file, file, -1))  # label -1 = different

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1_path, img2_path, label = self.pairs[idx]
        img1: torch.Tensor = transform(Image.open(img1_path).convert("RGB"))
        img2: torch.Tensor = transform(Image.open(img2_path).convert("RGB"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)
