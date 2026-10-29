
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import random
import os

dataset_dir="data/GTSRB/Train"
output_dir="results/deblurring"
os.makedirs(output_dir,exist_ok=True)

samples_list=[]

for class_name in sorted(os.listdir(dataset_dir)):
    class_path=os.path.join(dataset_dir,class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        if filename.lower().endswith((".png",".jpg",".jpeg",".ppm")):
            samples_list.append(
                (os.path.join(class_path,filename),class_name)
            )

random.seed(42)
samples=random.sample(samples_list,min(3,len(samples_list)))

fig,axes=plt.subplots(3,3,figsize=(12,10))

for row,(path,label) in enumerate(samples):
    original=np.array(Image.open(path).convert("RGB"))

    blurred=cv2.GaussianBlur(original,(0,0),2)

    smooth=cv2.GaussianBlur(original,(0,0),2)
    sharpened=cv2.addWeighted(original,2.0,smooth,-1.0,0)
    sharpened=np.clip(sharpened,0,255).astype(np.uint8)

    images=[
        ("Original",original),
        ("Gaussian Blur",blurred),
        ("Sharpened",sharpened)
    ]

    for col,(name,image) in enumerate(images):
        axes[row,col].imshow(image)
        axes[row,col].set_title(name)
        axes[row,col].axis("off")

        save_path=os.path.join(
            output_dir,f"sample_{row+1}_{name.lower().replace(' ','_')}.png"
        )
        Image.fromarray(image).save(save_path)

    axes[row,0].set_ylabel(f"Class {label}")

plt.tight_layout()
plt.savefig(os.path.join(output_dir,"comparison.png"),dpi=300,bbox_inches="tight")
plt.savefig(os.path.join(output_dir,"comparison.pdf"),bbox_inches="tight")
plt.show()
plt.close()

print("Results saved in:",output_dir)
