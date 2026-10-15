import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from dataset_loader import get_gtsrb_dataloaders
from corruptions import get_required_corruptions, apply_corruption

def extract_and_save_features():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    
    model_name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
   
    output_dir = "data/processed_features_gtsrb"
    os.makedirs(output_dir, exist_ok=True)
   

    train_loader, test_loader, classes = get_gtsrb_dataloaders()
   
    print("\n--- Extracting Training Features ---")
    train_features, train_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            
            outputs = model(images)
            
            features = outputs.last_hidden_state[:, 0, :] 
           
            train_features.append(features.cpu().numpy())
            train_labels.append(labels.numpy())
           
    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)
   
    np.save(os.path.join(output_dir, "train_features.npy"), train_features)
    np.save(os.path.join(output_dir, "train_labels.npy"), train_labels)
    print("Training features saved successfully!")

    print("\n--- Extracting Clean Test Features ---")
    test_features, test_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            features = outputs.last_hidden_state[:, 0, :]
           
            test_features.append(features.cpu().numpy())
            test_labels.append(labels.numpy())
           
    test_features = np.vstack(test_features)
    test_labels = np.concatenate(test_labels)
   
    np.save(os.path.join(output_dir, "val_features_clean.npy"), test_features)
    np.save(os.path.join(output_dir, "val_labels.npy"), test_labels)
    print("Clean test features saved successfully!")

if __name__ == "__main__":
    extract_and_save_features()