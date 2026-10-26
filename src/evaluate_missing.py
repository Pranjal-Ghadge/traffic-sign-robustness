
import csv
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corruptions import apply_corruption

BATCH_SIZE = 8
SEED = 42
NUM_IMAGES = 7842
CORRUPTION = "jpeg_compression"
SEVERITY = 5
RESULTS_FILE = "results/gtsrb_corruption_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Do not add a duplicate result
with open(RESULTS_FILE, "r", newline="") as f:
    rows = list(csv.DictReader(f))

if any(
    r["corruption"] == CORRUPTION
    and int(r["severity"]) == SEVERITY
    and int(r["num_images"]) == NUM_IMAGES
    for r in rows
):
    print("This experiment is already saved. Nothing to do.")
    raise SystemExit

dataset = ImageFolder("data/GTSRB/Train")
indices = np.arange(len(dataset))

_, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=dataset.targets
)
val_indices = val_indices[:NUM_IMAGES]

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dinov2 = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
dinov2.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std
    )
])

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

mlp = MLP().to(device)
mlp.load_state_dict(
    torch.load("models/gtsrb_mlp.pth", map_location=device, weights_only=True)
)
mlp.eval()

predictions = []
true_labels = []

for start in tqdm(range(0, len(val_indices), BATCH_SIZE)):
    images = []
    labels = []

    for idx in val_indices[start:start + BATCH_SIZE]:
        image, label = dataset[int(idx)]
        image = image.convert("RGB").resize(
            (224, 224), Image.Resampling.BILINEAR
        )

        corrupted = apply_corruption(
            np.array(image), CORRUPTION, SEVERITY
        )

        images.append(transform(Image.fromarray(corrupted)))
        labels.append(label)

    inputs = torch.stack(images).to(device)

    with torch.inference_mode():
        features = dinov2(inputs).last_hidden_state[:, 0, :]
        outputs = mlp(features)
        predicted = outputs.argmax(dim=1).cpu().tolist()

    predictions.extend(predicted)
    true_labels.extend(labels)

accuracy = accuracy_score(true_labels, predictions)
macro_f1 = f1_score(
    true_labels, predictions, average="macro", zero_division=0
)

print(f"JPEG compression severity 5")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1:.4f}")

with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["corruption", "severity", "num_images", "accuracy", "macro_f1"]
    )
    writer.writerow({
        "corruption": CORRUPTION,
        "severity": SEVERITY,
        "num_images": len(true_labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

print("Saved the missing result to:", RESULTS_FILE)
