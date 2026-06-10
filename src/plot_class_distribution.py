import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "data/raw/galaxy10.h5"
OUTPUT_PATH = "outputs/figures/class_distribution.png"

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
    labels = file["ans"][:]

class_ids, counts = np.unique(labels, return_counts=True)

plt.figure(figsize=(12, 6))
plt.bar(class_ids, counts)
plt.xticks(class_ids, CLASS_NAMES, rotation=45, ha="right")
plt.xlabel("Galaxy Class")
plt.ylabel("Number of Images")
plt.title("Galaxy10 SDSS Class Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.show()

print(f"Class distribution plot saved to {OUTPUT_PATH}")

print("\nClass counts:")
for class_id, count in zip(class_ids, counts):
    print(f"{class_id}: {CLASS_NAMES[int(class_id)]}: {count}")