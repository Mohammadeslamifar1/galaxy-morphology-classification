import torch
from model import create_model


model = create_model(num_classes=10)

dummy_images = torch.randn(32, 3, 69, 69)

outputs = model(dummy_images)

print(model)
print("Input shape:", dummy_images.shape)
print("Output shape:", outputs.shape)

if outputs.shape == torch.Size([32, 10]):
    print("Model is working correctly.")
else:
    print("Something is wrong with the output shape.")