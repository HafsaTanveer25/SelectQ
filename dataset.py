from typing import Sequence

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def build_datasets(root: str, image_size: int):
    train_tf, val_tf = build_transforms(image_size)
    train_set = datasets.ImageFolder(root=f"{root}/train", transform=train_tf)
    val_set = datasets.ImageFolder(root=f"{root}/val", transform=val_tf)
    return train_set, val_set


def build_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def make_subset(dataset: Dataset, indices: Sequence[int]) -> Subset:
    return Subset(dataset, list(indices))
