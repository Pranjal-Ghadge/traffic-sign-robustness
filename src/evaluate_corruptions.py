
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

from corruptions import apply_corruption

CORRUPTION = "clean"
SEVERITY = 1
NUM_IMAGES = 7842
BATCH_SIZE = 8
SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

dataset = ImageFolder(root="data/GTSRB/Train")

all_indices = np.arange(len(dataset))

_, val_indices = train_test_split(
    all_indices,
    test_size=0.2,
    random_state=SEED,
    stratify=dataset.targets
)

val_indices = val_indices[:NUM_IMAGES]

print("Validation images selected:", len(val_indices))

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-small"
)

dinov2 = AutoModel.from_pretrained(
    "facebook/dinov2-small"
).to(device)

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
    torch.load(
        "models/gtsrb_mlp.pth",
        map_location=device,
        weights_only=True
    )
)

mlp.eval()

predictions = []
true_labels = []

for start in tqdm(
    range(0, len(val_indices), BATCH_SIZE)
):

    batch_indices = val_indices[
        start:start + BATCH_SIZE
    ]

    images = []
    labels = []

    for idx in batch_indices:

        image, label = dataset[int(idx)]
        image = image.convert("RGB")

        if CORRUPTION == "clean":

            processed_image = image

        else:

            resized_image = image.resize(
                (224, 224),
                Image.Resampling.BILINEAR
            )

            image_np = np.array(resized_image)

            corrupted = apply_corruption(
                image_np,
                CORRUPTION,
                SEVERITY
            )

            processed_image = Image.fromarray(
                corrupted
            )

        images.append(
            transform(processed_image)
        )

        labels.append(label)

    inputs = torch.stack(images).to(device)

    with torch.inference_mode():

        features = dinov2(
            inputs
        ).last_hidden_state[:, 0, :]

        outputs = mlp(features)

        predicted = outputs.argmax(
            dim=1
        ).cpu().numpy()

    predictions.extend(
        predicted.tolist()
    )

    true_labels.extend(labels)

accuracy = accuracy_score(
    true_labels,
    predictions
)

macro_f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

print("\n--- Evaluation Results ---")
print("Condition:", CORRUPTION)

if CORRUPTION != "clean":
    print("Severity:", SEVERITY)

print("Images evaluated:", len(true_labels))
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1:.4f}")
