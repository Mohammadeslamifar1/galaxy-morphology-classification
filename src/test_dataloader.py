import torch
from dataset import create_dataloaders, CLASS_NAMES


train_loader, val_loader, test_loader = create_dataloaders(batch_size=32)

images, labels = next(iter(train_loader))

print("Number of classes:", len(CLASS_NAMES))
print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First labels:", labels[:10])

print("Dataloader is working.")