
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

feature_dir = "data/processed_features_gtsrb"

X_train = np.load(os.path.join(feature_dir, "train_features.npy"))
y_train = np.load(os.path.join(feature_dir, "train_labels.npy"))

X_val = np.load(os.path.join(feature_dir, "val_features_clean.npy"))
y_val = np.load(os.path.join(feature_dir, "val_labels.npy"))

assert X_train.shape == (31367, 384)
assert X_val.shape == (7842, 384)
assert set(np.unique(y_train)) == set(range(43))
assert set(np.unique(y_val)) == set(range(43))

train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).long()
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val).float(),
    torch.from_numpy(y_val).long()
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 43)
        )

    def forward(self, x):
        return self.network(x)


model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

best_val_loss = float("inf")
best_state = None
patience = 5
wait = 0

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

for epoch in range(20):
    model.train()
    running_loss = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(labels)

    train_loss = running_loss / len(train_dataset)

    model.eval()
    val_loss_total = 0.0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item() * len(labels)

    val_loss = val_loss_total / len(val_dataset)

    print(
        f"Epoch {epoch + 1:02d}/20 | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

model.load_state_dict(best_state)
torch.save(model.state_dict(), "models/gtsrb_mlp.pth")


def evaluate(loader):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features.to(device))
            predicted = outputs.argmax(dim=1).cpu().numpy()

            predictions.extend(predicted)
            true_labels.extend(labels.numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")

    return accuracy, f1, true_labels, predictions


train_acc, train_f1, _, _ = evaluate(train_loader)
val_acc, val_f1, y_true, y_pred = evaluate(val_loader)

print("\n--- Clean Baseline Results ---")
print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Training macro-F1: {train_f1:.4f}")
print(f"Validation macro-F1: {val_f1:.4f}")

cm = confusion_matrix(y_true, y_pred, labels=np.arange(43))

plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="Blues")
plt.title("GTSRB Clean Validation Confusion Matrix - DINOv2 + MLP")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(range(43), rotation=90, fontsize=6)
plt.yticks(range(43), fontsize=6)
plt.colorbar()
plt.tight_layout()
plt.savefig("results/gtsrb_mlp_confusion_matrix.pdf")
plt.close()

print("\nModel saved: models/gtsrb_mlp.pth")
print("Confusion matrix saved: results/gtsrb_mlp_confusion_matrix.pdf")
