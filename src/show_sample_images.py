
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image
import random
import os

dataset=ImageFolder("data/GTSRB/Train")

output_dir="results/deblurring"
os.makedirs(output_dir,exist_ok=True)

random.seed(42)
samples=random.sample(range(len(dataset)),3)

fig,axes=plt.subplots(3,3,figsize=(12,10))

for row,idx in enumerate(samples):
    path,label=dataset.samples[idx]

    original=np.array(Image.open(path).convert("RGB"))

    blurred=cv2.GaussianBlur(original,(0,0),2)

    smooth=cv2.GaussianBlur(original,(0,0),2)
    sharpened=cv2.addWeighted(original,2.0,smooth,-1.0,0)
    sharpened=np.clip(sharpened,0,255).astype(np.uint8)

    images=[
        ("Original image",original),
        ("Gaussian blur",blurred),
        ("Sharpened image",sharpened)
    ]

    for col,(name,image) in enumerate(images):
        axes[row,col].imshow(image)
        axes[row,col].set_title(name)
        axes[row,col].axis("off")

        save_path=os.path.join(
            output_dir,f"sample_{row+1}_{name.split()[0].lower()}.png"
        )
        Image.fromarray(image).save(save_path)

    axes[row,0].set_ylabel(f"Class {label}")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir,"comparison.png"),
    dpi=300,bbox_inches="tight"
)
plt.savefig(
    os.path.join(output_dir,"comparison.pdf"),
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Images and comparison figures saved in:",output_dir)
