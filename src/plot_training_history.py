import json
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY_PATH = "outputs/figures/training_history.json"
OUTPUT_PATH = "outputs/figures/training_curves.png"


Path("outputs/figures").mkdir(parents=True, exist_ok=True)


with open(HISTORY_PATH, "r") as file:
    history = json.load(file)


epochs = range(1, len(history["train_loss"]) + 1)


plt.figure(figsize=(10, 5))
plt.plot(epochs, history["train_loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/figures/loss_curve.png", dpi=200)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/figures/accuracy_curve.png", dpi=200)
plt.show()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.show()


print(f"Loss curve saved to outputs/figures/loss_curve.png")
print(f"Accuracy curve saved to outputs/figures/accuracy_curve.png")
print(f"Combined training curves saved to {OUTPUT_PATH}")