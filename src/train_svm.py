
import os
import joblib
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

feature_dir = "data/processed_features_gtsrb"

X_train = np.load(os.path.join(feature_dir, "train_features.npy"))
y_train = np.load(os.path.join(feature_dir, "train_labels.npy"))

X_val = np.load(os.path.join(feature_dir, "val_features_clean.npy"))
y_val = np.load(os.path.join(feature_dir, "val_labels.npy"))

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Training classes:", len(np.unique(y_train)))
print("Validation classes:", len(np.unique(y_val)))

model = LinearSVC(
    C=1.0,
    dual=False,
    max_iter=5000,
    random_state=42
)

print("\nTraining Linear SVM...")
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)

train_f1 = f1_score(y_train, train_pred, average="macro")
val_f1 = f1_score(y_val, val_pred, average="macro")

print("\n--- Clean Baseline Results ---")
print(f"Training accuracy: {train_acc * 100:.2f}%")
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print(f"Training macro-F1: {train_f1:.4f}")
print(f"Validation macro-F1: {val_f1:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gtsrb_linear_svm.joblib")

cm = confusion_matrix(y_val, val_pred, labels=np.arange(43))

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("GTSRB Clean Validation Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.colorbar()
plt.xticks(range(43), rotation=90, fontsize=6)
plt.yticks(range(43), fontsize=6)
plt.tight_layout()
plt.savefig("results/gtsrb_clean_confusion_matrix.pdf")
plt.close()

print("\nClassifier saved to models/gtsrb_linear_svm.joblib")
print("Confusion matrix saved to results/gtsrb_clean_confusion_matrix.pdf")
