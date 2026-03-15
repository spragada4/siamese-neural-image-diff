# scripts/train.py

# Third-party
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local
from siamese_diff.dataset import SiameseImageDataset
from siamese_diff.model import SiameseNetwork


def train_siamese_model(
    data_path: str = "data",
    batch_size: int = 2,
    epochs: int = 5,
    lr: float = 0.001,
) -> None:
    """Train Siamese network on image pairs."""

    dataset = SiameseImageDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SiameseNetwork()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for img1, img2, label in loader:
            img1 = img1.float()
            img2 = img2.float()
            label = label.float()

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved as model.pth")


if __name__ == "__main__":
    train_siamese_model()
