import torch
from PIL import Image
from torchvision import transforms

from src.dataset import CLASS_NAMES
from src.model import create_resnet18_model


MODEL_PATH = "models/resnet18_galaxy.pth"


def get_prediction_transform(image_size=224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_prediction_model(model_path=MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_resnet18_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def predict_image(image, model, device=None, top_k=3):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = get_prediction_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    top_probabilities, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    predictions = []

    for probability, class_index in zip(top_probabilities[0], top_indices[0]):
        predictions.append(
            {
                "class_id": int(class_index.cpu()),
                "class_name": CLASS_NAMES[int(class_index.cpu())],
                "confidence": float(probability.cpu()),
            }
        )

    return predictions