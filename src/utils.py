import json
import random
from pathlib import Path

import numpy as np
import torch


def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, output_path):
    create_directory(Path(output_path).parent)

    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)


def load_json(input_path):
    with open(input_path, "r") as file:
        return json.load(file)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)