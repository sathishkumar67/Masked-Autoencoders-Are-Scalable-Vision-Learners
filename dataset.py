import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, pos: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[pos]).convert("RGB"))
