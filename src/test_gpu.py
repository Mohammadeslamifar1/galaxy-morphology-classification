import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))

    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)

    print("GPU tensor test successful.")
else:
    print("CUDA is not available.")