
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor


DATASET_DIR = "data/GTSRB/Train"
SEED = 42
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 256


def get_gtsrb_dataloaders():
    """Create the stratified 80:20 GTSRB train/validation split and loaders."""
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        ),
    ])

    dataset = ImageFolder(root=DATASET_DIR, transform=transform)
    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=dataset.targets,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, dataset.classes


def generate_deblurring_samples():
    """Generate the separate Gaussian-blur/sharpening visual demonstration."""
    dataset_dir = DATASET_DIR
    output_dir = "results/deblurring"
    os.makedirs(output_dir, exist_ok=True)

    samples_list = []
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".ppm")):
                samples_list.append((os.path.join(class_path, filename), class_name))

    random.seed(SEED)
    samples = random.sample(samples_list, min(3, len(samples_list)))

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row, (path, label) in enumerate(samples):
        original = np.array(Image.open(path).convert("RGB"))
        blurred = cv2.GaussianBlur(original, (0, 0), 2)
        smooth = cv2.GaussianBlur(original, (0, 0), 2)
        sharpened = cv2.addWeighted(original, 2.0, smooth, -1.0, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        images = [
            ("Original", original),
            ("Gaussian Blur", blurred),
            ("Sharpened", sharpened),
        ]

        for col, (name, image) in enumerate(images):
            axes[row, col].imshow(image)
            axes[row, col].set_title(name)
            axes[row, col].axis("off")

            save_path = os.path.join(
                output_dir,
                f"sample_{row + 1}_{name.lower().replace(' ', '_')}.png",
            )
            Image.fromarray(image).save(save_path)

        axes[row, 0].set_ylabel(f"Class {label}")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(os.path.join(output_dir, "comparison.pdf"), bbox_inches="tight")
    plt.close()

    print("Results saved in:", output_dir)


if __name__ == "__main__":
    train_loader, val_loader, classes = get_gtsrb_dataloaders()
    print("Training classes:", len(classes))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
