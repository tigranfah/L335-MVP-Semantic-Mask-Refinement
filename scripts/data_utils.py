import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import random
import numpy as np


def get_loaders(image_size=256, batch_size=32, data_split=(0.8, 0.1, 0.1), num_workers=2, root="./data",):
    """
    Creates and returns DataLoaders for the Oxford-IIIT Pet dataset.
    Splits dataset into Train / Refiner / Dev / Test.

    Args:
        image_size: image resolution
        batch_size: batch size for DataLoaders
        data_split: tuple of 4 floats (train, refiner, dev, test) summing to 1
        num_workers: DataLoader worker count
        root: dataset directory

    Returns:
        dict of DataLoaders:
            {
                'train': ...,
                'dev': ...,
                'test': ...,
            }
    """

    assert abs(sum(data_split) - 1.0) == 0, "Splits must sum to 1.0"

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        lambda x: ((x == 1) | (x == 3)).float(),  # binary pet mask
        transforms.Normalize((0.5,), (0.5,))  # [0,1] → [-1,1]
    ])

    # Dataset
    dataset = OxfordIIITPet(
        root=root,
        split="trainval",
        download=True,
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform,
    )

    # Indices and splits
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    total_len = len(indices)

    train_end = int(data_split[0] * total_len)
    dev_end = int((data_split[0] + data_split[1]) * total_len)

    train_idx = indices[:train_end]
    dev_idx = indices[train_end:dev_end]
    test_idx = indices[dev_end:]

    print(f"Total samples: {total_len}")
    print(f"Train: {len(train_idx)} | Dev: {len(dev_idx)} | Test: {len(test_idx)}")

    # Subsets
    train_dataset = Subset(dataset, train_idx)
    dev_dataset = Subset(dataset, dev_idx)
    test_dataset = Subset(dataset, test_idx)

    # DataLoaders
    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "dev": DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders
