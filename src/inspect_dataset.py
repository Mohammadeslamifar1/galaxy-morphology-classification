import h5py
import numpy as np

DATA_PATH = "data/raw/galaxy10.h5"

CLASS_NAMES = [
    "Disk, Face On, No Spiral",
    "Smooth, Completely Round",
    "Smooth, In Between Round",
    "Smooth, Cigar Shaped",
    "Disk, Edge On, Rounded Bulge",
    "Disk, Edge On, Boxy Bulge",
    "Disk, Edge On, No Bulge",
    "Disk, Face On, Tight Spiral",
    "Disk, Face On, Medium Spiral",
    "Disk, Face On, Loose Spiral",
]


with h5py.File(DATA_PATH, "r") as file:
    print("Dataset keys:", list(file.keys()))

    images = file["images"]
    labels = file["ans"]

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    unique_labels, counts = np.unique(labels, return_counts=True)

    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {CLASS_NAMES[int(label)]}  count={count}")