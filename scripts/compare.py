# scripts/compare.py

# Third-party
import torch
from PIL import Image
from torchvision import transforms

# Local
from siamese_diff.model import SiameseNetwork

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def compare_images(img1_path: str, img2_path: str) -> None:
    model = SiameseNetwork()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    with torch.no_grad():
        output1, output2 = model(img1, img2)

    distance = torch.nn.functional.pairwise_distance(output1, output2)
    print("Similarity distance:", distance.item())


if __name__ == "__main__":
    compare_images(
        "data/same/img1.png",
        "data/same/img2.png",
    )
