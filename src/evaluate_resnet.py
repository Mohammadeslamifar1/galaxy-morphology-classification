import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from dataset import create_dataloaders, CLASS_NAMES
from model import create_resnet18_model


MODEL_PATH = "models/resnet18_galaxy.pth"
CONFUSION_MATRIX_PATH = "outputs/figures/resnet18_confusion_matrix.png"
REPORT_PATH = "outputs/figures/resnet18_classification_report.txt"
METRICS_PATH = "outputs/figures/resnet18_test_metrics.json"


def evaluate_model(model, test_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("ResNet18 Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(
        batch_size=16,
        image_size=224,
    )

    model = create_resnet18_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)

    true_labels, predicted_labels = evaluate_model(model, test_loader, device)

    test_accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print(f"ResNet18 Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_PATH)

    with open(REPORT_PATH, "w") as file:
        file.write(f"ResNet18 Test Accuracy: {test_accuracy:.4f}\n\n")
        file.write(report)

    metrics = {
        "resnet18_test_accuracy": float(test_accuracy),
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
    print(f"Classification report saved to {REPORT_PATH}")
    print(f"Test metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()