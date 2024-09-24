from dataset import EyecandiesDataset
from constants import CLASSES
import numpy as np
import torch
import shutil
import os
from tqdm import tqdm


def calculate_mse(imageA: torch.Tensor, imageB: torch.Tensor) -> float:
    # Ensure the images have the same dimensions
    if imageA.shape != imageB.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Compute the MSE
    mse = torch.mean((imageA - imageB) ** 2).item()
    return mse


mses = {}
paths = {}
for class_name in CLASSES:
    dataset = EyecandiesDataset(f"../eyecandies-dataset/{class_name}", "train", "all", True, 256, 256)
    mses[class_name] = []
    paths[class_name] = []

    for imgs, p in dataset:
        original = imgs[-1]
        reconstruction = imgs[-2]

        mse = calculate_mse(original, reconstruction)

        mses[class_name].append(mse)
        paths[class_name].append([p[-1], p[-2]])

    print(f"Finished calculating MSE for {class_name}")


for class_name, class_mses in mses.items():
    print(f"Class: {class_name}")
    print(f"Mean MSE: {np.mean(class_mses)}")
    print(f"Standard Deviation: {np.std(class_mses)}")

    print(f"Max MSE: {max(class_mses)}, Position: {class_mses.index(max(class_mses))}")
    print(f"Min MSE: {min(class_mses)}, Position: {class_mses.index(min(class_mses))}")

    # now take the 10 highest and 10 lowest MSEs and get their positions
    sorted_mses = sorted(class_mses)
    highest_mses = sorted_mses[-3:]
    lowest_mses = sorted_mses[:3]
    highest_positions = [class_mses.index(mse) for mse in highest_mses]
    lowest_positions = [class_mses.index(mse) for mse in lowest_mses]

    os.makedirs(f"highest_mses/{class_name}", exist_ok=True)
    for i, position in enumerate(highest_positions):
        original_path, reconstructed_path = paths[class_name][position]
        shutil.copy(original_path, f"highest_mses/{class_name}/original_{i}.png")
        shutil.copy(reconstructed_path, f"highest_mses/{class_name}/reconstructed_{i}.png")

    os.makedirs(f"lowest_mses/{class_name}", exist_ok=True)
    for i, position in enumerate(lowest_positions):
        original_path, reconstructed_path = paths[class_name][position]
        shutil.copy(original_path, f"lowest_mses/{class_name}/original_{i}.png")
        shutil.copy(reconstructed_path, f"lowest_mses/{class_name}/reconstructed_{i}.png")
