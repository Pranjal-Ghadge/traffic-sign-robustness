
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

feature_dir="data/processed_features_gtsrb"

X_val=np.load(os.path.join(feature_dir,"val_features_clean.npy"))
y_val=np.load(os.path.join(feature_dir,"val_labels.npy"))

model=joblib.load("models/gtsrb_linear_svm.joblib")
val_pred=model.predict(X_val)

cm=confusion_matrix(y_val,val_pred,labels=np.arange(43))

plt.figure(figsize=(12,10))
plt.imshow(cm,interpolation="nearest",cmap="Blues")
plt.title("GTSRB Clean Validation Confusion Matrix - Linear SVM")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.colorbar()
plt.xticks(range(43),rotation=90,fontsize=6)
plt.yticks(range(43),fontsize=6)
plt.tight_layout()

os.makedirs("results",exist_ok=True)
plt.savefig("results/gtsrb_clean_confusion_matrix.pdf")
plt.close()

print("SVM confusion matrix regenerated successfully.")
print("Saved to results/gtsrb_clean_confusion_matrix.pdf")
