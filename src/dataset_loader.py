import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from transformers import AutoImageProcessor

def get_gtsrb_dataloaders(data_dir="data/GTSRB", batch_size=32, train_split=0.8):
    """
    Loads GTSRB dataset, applies DINOv2 preprocessing, splits into train/test, 
    and returns PyTorch DataLoaders.
    """
    model_name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # DINOv2 expects standard image transformations handled by the processor,
    # but we can wrap it using standard torchvision transforms or process on-the-fly.
    # Here we use standard torchvision resizing/normalization matching DINOv2 stats:
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean, 
            std=processor.image_std
        ),
    ])
    
    # Load dataset using ImageFolder (assumes structure: data/GTSRB/train/class_folders/images)
    # If your GTSRB folder contains subfolders for classes directly, point to that folder.
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path {data_dir} not found. Please check your folder structure.")
        
    dataset = ImageFolder(root=data_dir, transform=transform)
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded successfully!")
    print(f"Total classes: {len(dataset.classes)}")
    print(f"Training samples: {train_size}, Testing samples: {test_size}")
    
    return train_loader, test_loader, dataset.classes

if __name__ == "__main__":
    # Test the loader script
    train_loader, test_loader, classes = get_gtsrb_dataloaders()