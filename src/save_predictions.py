from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dataset import create_dataloaders, CLASS_NAMES
from model import create_model


MODEL_PATH = "models/baseline_cnn.pth"
OUTPUT_PATH = "outputs/predictions/sample_predictions.png"


def denormalize_image(tensor_image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = tensor_image.cpu() * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()

    return image


def main():
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(batch_size=32)

    model = create_model(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    images, labels = next(iter(test_loader))

    images_device = images.to(device)

    with torch.no_grad():
        outputs = model(images_device)
        probabilities = torch.softmax(outputs, dim=1)
        confidence_scores, predictions = torch.max(probabilities, 1)

    plt.figure(figsize=(16, 10))

    for index in range(12):
        image = denormalize_image(images[index])
        true_label = int(labels[index])
        predicted_label = int(predictions[index].cpu())
        confidence = float(confidence_scores[index].cpu())

        title_color = "green" if true_label == predicted_label else "red"

        plt.subplot(3, 4, index + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"True: {CLASS_NAMES[true_label]}\n"
            f"Pred: {CLASS_NAMES[predicted_label]}\n"
            f"Conf: {confidence:.2f}",
            fontsize=8,
            color=title_color,
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.show()

    print(f"Sample predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()