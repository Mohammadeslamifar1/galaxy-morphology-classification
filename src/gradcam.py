import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.predict import get_prediction_transform


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activations
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradients
        )

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class_index):
        self.model.zero_grad()

        output = self.model(input_tensor)

        score = output[:, target_class_index]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()

        if cam.max() != 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def create_gradcam_overlay(image, model, device, target_class_index):
    image = image.convert("RGB")

    transform = get_prediction_transform(image_size=224)
    input_tensor = transform(image).unsqueeze(0).to(device)

    target_layer = model.layer4[-1]

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor, target_class_index)
    gradcam.remove_hooks()

    original_image = np.array(image)
    height, width, _ = original_image.shape

    cam_resized = cv2.resize(cam, (width, height))

    heatmap = plt.get_cmap("jet")(cam_resized)
    heatmap = np.delete(heatmap, 3, axis=2)
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = 0.55 * original_image + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay