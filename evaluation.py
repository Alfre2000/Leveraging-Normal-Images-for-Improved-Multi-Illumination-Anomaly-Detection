import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from typing import Literal, List, Dict
from dataset import EyecandiesDataset
from ad_types import Method
import pandas as pd
from torch.utils.data import DataLoader
from constants import CLASSES
import os
from architectures.dinomaly.load import load_model as load_dino_model
from architectures.rd4ad.load import load_model as load_rd4ad_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def performance_table(
    method: Method,
    architecture: str,
    dataset_path: str,
    grouped: bool = False,
    size: int = 256,
    isize: int = 392,
    zero_shot: bool = False,
) -> pd.DataFrame:
    """
    Generate a performance table for the given model and method.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - method (Method): The method used to generate the anomaly maps.
    - dataset_path (str): The path to the dataset.
    - grouped (bool, optional): If True, images will be grouped together based on their sequence. Default is False.
    - size (int, optional): The size to which the images will be resized. Default is 256.
    - isize (int, optional): Another dimension for image resizing, used in transformations. Default is 392.

    Returns:
    - pd.DataFrame: The performance table as a pandas DataFrame.
    """
    maps_combination = "same_weights" if method in ["rgb_normal", "rgb_normal_real"] else "average"
    results = []

    classes = ["first", "second"] if zero_shot else CLASSES
    for class_name in classes:
        ckp_path = f'./checkpoints/dino/{"fusion" if grouped else "avg"}/{"zero_shot/" if zero_shot else ""}{method}/{class_name}.pth'
        model = load_model(architecture, ckp_path, method, grouped)
        if zero_shot:
            test_classes = CLASSES[5:] if class_name == "first" else CLASSES[:5]
            path = [f"{dataset_path}/{c}" for c in test_classes]
        else:
            path = f"{dataset_path}/{class_name}"
        test_data = EyecandiesDataset(
            root=path,
            phase="test",
            method=method,
            grouped=grouped,
            size=size,
            isize=isize,
        )
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        anomalies_set = [
            {"values": ["anomalous_colors"], "name": "colors"},
            {"values": ["anomalous_bumps", "anomalous_dents", "anomalous_normals"], "name": "structural"},
            {"values": ["anomalous_bumps", "anomalous_dents", "anomalous_normals", "anomalous_colors"], "name": "all"}
        ]

        for anomalies in anomalies_set:
            metrics = evaluation(
                model,
                test_dataloader,
                anomalies=anomalies["values"],
                grouped=grouped,
                method=maps_combination
            )

            results.append({
                'Category': class_name,
                'Anomalies': anomalies["name"],
                **metrics
            })

    results = pd.DataFrame(results)
    performance_dir = f'./performance/dino/{"fusion" if grouped else "avg"}/{"zero_shot/" if zero_shot else ""}'
    os.makedirs(performance_dir, exist_ok=True)
    results.to_csv(f'{performance_dir}{method}.csv', index=False)
    return results


def evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    anomalies: list = ["anomalous_bumps", "anomalous_dents", "anomalous_colors", "anomalous_normals"],
    grouped: bool = False,
    n_tresh: int = 50,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate the performance of an anomaly detection model on a given dataset.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - dataloader (DataLoader): The DataLoader object for the dataset.
    - anomalies (list, optional): The list of anomalies to consider. Default is all anomalies.
    - grouped (bool, optional): If True, treats input_images as a grouped list of lists of tensors. Default is False.
    - **kwargs: Additional keyword arguments to pass to the predict function.

    Returns:
    - Tuple[float, float, float]: The AUROC scores at the pixel-level and image-level.
    """
    model.eval()

    gt_list_pixel = []
    pr_list_pixel = []
    gt_list_image = []
    pr_list_image = []
    thresholds = np.linspace(0, 1, num=n_tresh)

    with torch.no_grad():
        for imgs, ground_truth, label, types in dataloader:
            if label.item() == 1 and any(t[0] not in anomalies for t in types):
                continue

            anomaly_map = predict(model, imgs, grouped=grouped, **kwargs)

            anomaly_map = anomaly_map[68:324, 68:324]
            ground_truth = ground_truth[:, :, 68:324, 68:324]

            # Process ground truth
            ground_truth = (ground_truth > 0.5).int()
            gt_list_pixel.extend(ground_truth.cpu().numpy().flatten())
            pr_list_pixel.extend(anomaly_map.flatten())

            # Store the maximum value for image-level evaluation
            gt_list_image.append(ground_truth.max().item())
            pr_list_image.append(anomaly_map.max().item())

    # Pixel-level metrics
    auroc_px = round(roc_auc_score(gt_list_pixel, pr_list_pixel), 3)
    avg_precision_px = round(average_precision_score(gt_list_pixel, pr_list_pixel), 3)
    f1_max_px = round(max(f1_score(gt_list_pixel, pr_list_pixel >= t) for t in thresholds), 3)

    # Image-level metrics
    auroc_image = round(roc_auc_score(gt_list_image, pr_list_image), 3)
    avg_precision_image = round(average_precision_score(gt_list_image, pr_list_image), 3)
    f1_max_image = round(max(f1_score(gt_list_image, [p >= t for p in pr_list_image]) for t in thresholds), 3)

    return {
        'Pixel AUROC': auroc_px,
        'Pixel F1 Max': f1_max_px,
        'Pixel Average Precision': avg_precision_px,
        'Sample AUROC': auroc_image,
        'Sample F1 Max': f1_max_image,
        'Sample Average Precision': avg_precision_image
    }


def predict(
    model: torch.nn.Module,
    input_images: List[torch.Tensor],
    method: Literal["average", "same_weights"] = "average",
    sigma: int = 4,
    grouped: bool = False,
) -> np.ndarray:
    """
    Predict anomaly maps for a given set of input images using a specified model.

    Parameters:
    model (torch.nn.Module): The model to use for prediction.
    input_images (List[torch.Tensor]): A list of input images as PyTorch tensors.
    method (Literal["average", "same_weights"], optional): The method to use for combining anomaly maps.
        "average" computes the mean of anomaly maps, while "same_weights" assigns half the total weight
        to the last image. Default is "average".
    sigma (int, optional): The standard deviation for Gaussian filtering of the anomaly maps. Default is 4.
    grouped (bool, optional): If True, treats input_images as a grouped list of lists of tensors. Default is False.

    Returns:
    np.ndarray: The final combined anomaly map.
    """

    input_images = [input_images] if grouped else input_images

    anomaly_maps = []
    for img in input_images:
        if grouped:
            img = [i.to(device) for i in img]
            out_size = img[0].shape[-1]
        else:
            out_size = img.shape[-1]
            img = img.to(device)

        inputs, outputs = model(img)

        anomaly_map = calculate_anomaly_map(inputs, outputs, out_size, mode="add")
        anomaly_map = gaussian_filter(anomaly_map, sigma=sigma)
        anomaly_maps.append(anomaly_map)


    anomaly_maps = np.array(anomaly_maps)

    # Mean of anomaly maps
    if method != "same_weights":
        anomaly_map = np.mean(anomaly_maps, axis=0)
    else:
        # the last image is the normal image and should the same weight as the other images combined
        if len(anomaly_maps) > 1:
            weights = np.ones(len(anomaly_maps))
            weights[:-1] = 0.5 / (
                len(anomaly_maps) - 1
            )  # Weight for the first N-1 images
            weights[-1] = 0.5  # Weight for the last image
            anomaly_map = np.average(anomaly_maps, axis=0, weights=weights)
        else:
            anomaly_map = anomaly_maps[0]

    return anomaly_map


def calculate_anomaly_map(
    feature_maps_source: List[torch.Tensor],
    feature_maps_target: List[torch.Tensor],
    out_size: int,
    mode: Literal["mul", "add"] = "add",
) -> np.ndarray:
    """
    Calculate the anomaly map between two lists of feature tensors.

    This function compares corresponding feature tensors to generate an anomaly map.
    The anomaly is calculated using 1 minus the cosine similarity between the corresponding features.
    The final anomaly map is obtained by either multiplying or adding these individual maps based on the mode.

    Parameters:
    - feature_maps_source (List[torch.Tensor]): The list of feature tensors from the source.
    - feature_maps_target (List[torch.Tensor]): The list of feature tensors from the target.
    - out_size (int): The output size to which each anomaly map will be resized.
    - mode (Literal['mul', 'add']): The mode of combining individual anomaly maps
                                    ('mul' for multiplication, 'add' for addition).

    Returns:
    - np.ndarray: The combined anomaly map as a numpy array.
    """
    # Initialize the anomaly map as ones or zeros based on the mode
    anomaly_map = (
        np.ones((out_size, out_size))
        if mode == "mul"
        else np.zeros((out_size, out_size))
    )

    for fs, ft in zip(feature_maps_source, feature_maps_target):
        # Calculate anomaly map for each feature pair
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(
            a_map, size=(out_size, out_size), mode="bilinear", align_corners=True
        )
        a_map = a_map[0, 0, :, :].cpu().detach().numpy()

        # Update the global anomaly map
        if mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map


def load_model(architecture, ckp_path, method, grouped):
    if architecture == "dino":
        return load_dino_model(ckp_path, method, grouped)
    elif architecture == "rd4ad":
        return load_rd4ad_model(ckp_path, method, grouped)
