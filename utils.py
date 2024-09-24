import torch
import numpy as np
import random
from typing import List
import cv2


def setup_seed(seed: int):
    """
    Set the seed to ensure reproducibility. This function sets the seed for random number
    generators in torch, numpy, and the built-in random module, ensuring that all stochastic
    operations will produce the same results given the same seed. It also makes the behavior
    of convolutions in torch deterministic.

    Parameters:
    seed (int): The seed value to be used for all random number generators.

    Note:
    - Setting `torch.backends.cudnn.deterministic` to True can impact performance negatively
      and might lead to a slight slowdown.
    - Disabling `torch.backends.cudnn.benchmark` prevents cuDNN from using its auto-tuner
      to find the best algorithms for your particular configurations, which can also
      degrade performance. However, it's necessary for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_similarity_loss(tensor_list_a: List[torch.Tensor], tensor_list_b: List[torch.Tensor]):
    """
    Compute the cosine similarity loss between corresponding tensors in two lists.

    The loss is calculated as the mean of 1 minus the cosine similarity of each pair of tensors.
    It effectively measures the average angle difference between pairs of tensors in the multi-dimensional space.

    Parameters:
    tensor_list_a (List[torch.Tensor]): The first list of tensors.
    tensor_list_b (List[torch.Tensor]): The second list of tensors, must be the same length as tensor_list_a.

    Returns:
    torch.Tensor: A single tensor representing the average cosine similarity loss between the tensor pairs.
    """
    cos_loss_fn = torch.nn.CosineSimilarity()

    # Validate input lists are not empty and have the same length
    if not tensor_list_a or not tensor_list_b or len(tensor_list_a) != len(tensor_list_b):
        raise ValueError("Input tensor lists must be non-empty and of the same length.")

    # Initialize the loss as a zero tensor
    loss = torch.zeros(1, device=tensor_list_a[0].device)

    # Compute the cosine similarity loss for each pair of tensors
    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        # Reshape the tensors to 2D (batch_size, -1)
        tensor_a = tensor_a.view(tensor_a.shape[0], -1)
        tensor_b = tensor_b.view(tensor_b.shape[0], -1)

        loss += torch.mean(1 - cos_loss_fn(tensor_a, tensor_b))

    return loss


def count_parameters(model: torch.nn.Module) -> int:
    """
    Calculate the total number of trainable parameters in a PyTorch model.
    This function iterates through all parameters of the given model, checks if they require gradients
    (indicating they are trainable), and sums up their total number of elements.

    Parameters:
    model (torch.nn.Module): The model whose parameters are to be counted.

    Returns:
    int: The total number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params



def min_max_norm(images: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Perform min-max normalization on a single image tensor or a list of image tensors, normalizing them together
    if it's a list.

    This function normalizes the pixel values of the image(s) to be within the range [0, 1].
    If a list of images is provided, normalization is based on the absolute minimum and maximum pixel
    values found among all images in the list. It subtracts the minimum value (found across all images
    if a list is provided) from all pixels in each image and then divides by the range of the pixel
    values (max - min or absolute max - absolute min if a list is provided).

    Parameters:
    - images (list[torch.Tensor] | torch.Tensor): A single image tensor or a list of image tensors to be normalized.

    Returns:
    - list[torch.Tensor] | torch.Tensor: The normalized image tensor(s) with pixel values in the range [0, 1].
    """
    # Handle the case where the input is a single image tensor
    if isinstance(images, np.ndarray):
        a_min, a_max = images.min(), images.max()
        return (images - a_min) / (a_max - a_min)

    # Handle the case where the input is a list of image tensors
    elif isinstance(images, list) and images:
        abs_min = min(image.min() for image in images)
        abs_max = max(image.max() for image in images)
        return [(image - abs_min) / (abs_max - abs_min) for image in images]

    return images


def show_cam_on_image(img: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    """
    Overlay an anomaly map onto an image for visualization.

    This function combines an anomaly map with an image, normalizes the combined result,
    and converts it to an 8-bit format. The result highlights areas of interest or anomalies
    on the original image.

    Parameters:
    - img (np.ndarray): The original image as a NumPy array.
    - anomaly_map (np.ndarray): The anomaly map as a NumPy array, indicating areas of interest.

    Returns:
    - np.ndarray: The image with the anomaly map overlaid, converted to 8-bit format.
    """
    # Normalize the anomaly map and the image and combine them
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    # Normalize the combined image to have a maximum value of 1
    cam = cam / np.max(cam)
    # Convert the normalized image to an 8-bit format
    return np.uint8(255 * cam)


def cvt2heatmap(gray: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to a heatmap using the JET colormap.

    This function applies a color mapping transformation to convert a single-channel
    grayscale image into a three-channel heatmap, enhancing visual perception of the
    image's details.

    Parameters:
    - gray (np.ndarray): A single-channel grayscale image as a NumPy array.

    Returns:
    - np.ndarray: The converted heatmap as a three-channel image in BGR format.
    """
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
