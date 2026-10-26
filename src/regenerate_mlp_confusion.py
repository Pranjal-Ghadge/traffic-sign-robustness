
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dir="data/processed_features_gtsrb"

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
model.load_state_dict(
    torch.load("models/gtsrb_mlp.pth",map_location=device)
)
model.eval()

predictions=[]

with torch.no_grad():
    for start in range(0,len(X_val),256):
        features=torch.from_numpy(
            X_val[start:start+256]
        ).float().to(device)

        outputs=model(features)
        predictions.extend(
            outputs.argmax(dim=1).cpu().numpy()
        )

accuracy=accuracy_score(y_val,predictions)
macro_f1=f1_score(y_val,predictions,average="macro")
cm=confusion_matrix(y_val,predictions,labels=np.arange(43))

print(f"Validation accuracy: {accuracy*100:.2f}%")
print(f"Validation macro-F1: {macro_f1:.4f}")

os.makedirs("results",exist_ok=True)

plt.figure(figsize=(12,10))
plt.imshow(cm,interpolation="nearest",cmap="Blues")
plt.title("GTSRB Clean Validation Confusion Matrix - DINOv2 + MLP")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(range(43),rotation=90,fontsize=6)
plt.yticks(range(43),fontsize=6)
plt.colorbar()
plt.tight_layout()
plt.savefig("results/gtsrb_mlp_confusion_matrix.pdf")
plt.close()

print("Updated: results/gtsrb_mlp_confusion_matrix.pdf")
