
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

feature_dir="data/processed_features_gtsrb"

X_train=np.load(os.path.join(feature_dir,"train_features.npy"))
y_train=np.load(os.path.join(feature_dir,"train_labels.npy"))
X_val=np.load(os.path.join(feature_dir,"val_features_clean.npy"))
y_val=np.load(os.path.join(feature_dir,"val_labels.npy"))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(384,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,43)
        )

    def forward(self,x):
        return self.network(x)

model=MLP().to(device)
model.load_state_dict(torch.load("models/gtsrb_mlp.pth",map_location=device))
model.eval()

def evaluate(X,y):
    dataset=TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long()
    )
    loader=DataLoader(dataset,batch_size=256,shuffle=False)
    actual=[]
    predicted=[]

    with torch.no_grad():
        for features,labels in loader:
            outputs=model(features.to(device))
            preds=outputs.argmax(dim=1).cpu().numpy()
            predicted.extend(preds)
            actual.extend(labels.numpy())

    accuracy=accuracy_score(actual,predicted)
    macro_f1=f1_score(actual,predicted,average="macro")
    return accuracy,macro_f1

train_acc,train_f1=evaluate(X_train,y_train)
val_acc,val_f1=evaluate(X_val,y_val)

print("\n--- Clean Baseline Results ---")
print(f"Training accuracy: {train_acc*100:.2f}%")
print(f"Validation accuracy: {val_acc*100:.2f}%")
print(f"Training macro-F1: {train_f1:.4f}")
print(f"Validation macro-F1: {val_f1:.4f}")
