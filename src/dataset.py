import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATA_PATH = "data/raw/galaxy10.h5"


CLASS_NAMES = [
    "Disk Face On No Spiral",
    "Smooth Completely Round",
    "Smooth In Between Round",
    "Smooth Cigar Shaped",
    "Disk Edge On Rounded Bulge",
    "Disk Edge On Boxy Bulge",
    "Disk Edge On No Bulge",
    "Disk Face On Tight Spiral",
    "Disk Face On Medium Spiral",
    "Disk Face On Loose Spiral",
]


class Galaxy10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return image, label


def load_galaxy10_data(data_path=DATA_PATH):
    with h5py.File(data_path, "r") as file:
        images = np.array(file["images"])
        labels = np.array(file["ans"])

    return images, labels


def create_data_splits(images, labels, test_size=0.15, val_size=0.15, random_state=42):
    train_val_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    train_val_labels = labels[train_val_indices]

    adjusted_val_size = val_size / (1.0 - test_size)

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    return train_indices, val_indices, test_indices


def get_transforms(image_size=69):
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def create_dataloaders(batch_size=64, num_workers=0):
    images, labels = load_galaxy10_data()

    train_indices, val_indices, test_indices = create_data_splits(images, labels)

    train_transform, eval_transform = get_transforms()

    train_dataset = Galaxy10Dataset(
        images=images[train_indices],
        labels=labels[train_indices],
        transform=train_transform,
    )

    val_dataset = Galaxy10Dataset(
        images=images[val_indices],
        labels=labels[val_indices],
        transform=eval_transform,
    )

    test_dataset = Galaxy10Dataset(
        images=images[test_indices],
        labels=labels[test_indices],
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader