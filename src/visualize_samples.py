import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_PATH = "data/raw/galaxy10.h5"
OUTPUT_PATH = "outputs/figures/sample_galaxies.png"

CLASS_NAMES = [
    "Disk Face On No Spiral",
    "Smooth Completely Round",
    "Smooth In Between Round",
    "Smooth Cigar Shaped",
    "Disk Edge On Rounded Bulge",
    "Disk Edge On Boxy Bulge",
    "Disk Edge On No Bulge",
    "Disk Face On Tight Spiral",
    "Disk Face On Medium Spiral",
    "Disk Face On Loose Spiral",
]

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

with h5py.File(DATA_PATH, "r") as file:
    images = file["images"][:]
    labels = file["ans"][:]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for class_id in range(10):
    class_indices = np.where(labels == class_id)[0]
    sample_index = class_indices[0]

    image = images[sample_index]

    axes[class_id].imshow(image)
    axes[class_id].set_title(CLASS_NAMES[class_id], fontsize=8)
    axes[class_id].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.show()

print(f"Sample image grid saved to {OUTPUT_PATH}")