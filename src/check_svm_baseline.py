
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

feature_dir="data/processed_features_gtsrb"

X_train=np.load(os.path.join(feature_dir,"train_features.npy"))
y_train=np.load(os.path.join(feature_dir,"train_labels.npy"))
X_val=np.load(os.path.join(feature_dir,"val_features_clean.npy"))
y_val=np.load(os.path.join(feature_dir,"val_labels.npy"))

model=joblib.load("models/gtsrb_linear_svm.joblib")

train_pred=model.predict(X_train)
val_pred=model.predict(X_val)

print("--- Linear SVM Clean Baseline ---")
print(f"Training accuracy: {accuracy_score(y_train,train_pred)*100:.2f}%")
print(f"Validation accuracy: {accuracy_score(y_val,val_pred)*100:.2f}%")
print(f"Training macro-F1: {f1_score(y_train,train_pred,average='macro'):.4f}")
print(f"Validation macro-F1: {f1_score(y_val,val_pred,average='macro'):.4f}")
