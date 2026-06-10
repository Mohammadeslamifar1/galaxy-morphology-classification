import torch
import torchvision
import numpy as np
import h5py
import sklearn
import streamlit

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Environment is ready.")