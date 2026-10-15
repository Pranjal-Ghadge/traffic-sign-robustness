
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor
from sklearn.model_selection import train_test_split


def get_gtsrb_dataloaders(
    data_dir="data/GTSRB",
    batch_size=32,
    train_split=0.8
):
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])

    train_dir = os.path.join(data_dir, "Train")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training folder not found: {train_dir}"
        )

    dataset = ImageFolder(
        root=train_dir,
        transform=transform
    )
    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_split,
        random_state=42,
        stratify=dataset.targets
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print("Dataset loaded successfully!")
    print(f"Total classes: {len(dataset.classes)}")
    print(f"Total images: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader, dataset.classes


if __name__ == "__main__":
    get_gtsrb_dataloaders()
