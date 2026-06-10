import json
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY_PATH = "outputs/figures/resnet18_training_history.json"
OUTPUT_PATH = "outputs/figures/resnet18_training_curves.png"


Path("outputs/figures").mkdir(parents=True, exist_ok=True)


with open(HISTORY_PATH, "r") as file:
    history = json.load(file)


epochs = range(1, len(history["train_loss"]) + 1)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 Loss Curve")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet18 Accuracy Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.show()


print(f"ResNet18 training curves saved to {OUTPUT_PATH}")