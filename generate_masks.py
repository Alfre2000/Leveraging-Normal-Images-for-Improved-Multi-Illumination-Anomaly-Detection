from transformers import AutoModelForImageSegmentation
import torch
import torch.nn.functional as F
import cv2
from skimage import io
import numpy as np
from dataset import EyecandiesDataset
from constants import CLASSES
from tqdm import tqdm as ProgressBar


def preprocess_image(im: np.ndarray, model_input_size: list, division: int = 20) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = torch.divide(image, division)
    return image


def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
model.to(device)

for class_name in ["Confetto"]:
    train_dataset = EyecandiesDataset(f"../eyecandies-dataset/{class_name}", "train", "rgb_normal", True, 256, 256)
    test_dataset = EyecandiesDataset(f"../eyecandies-dataset/{class_name}", "test", "rgb_normal", True, 256, 256)
    datasets = [train_dataset, test_dataset]

    for dataset in datasets:
        for img, p, *_ in ProgressBar(dataset, desc=f"Generating masks for {class_name} {dataset.phase}"):
            # prepare input
            image_path = p[0]
            orig_im = io.imread(image_path)
            orig_im_size = orig_im.shape[0:2]

            division = 30
            while True:
                image = preprocess_image(orig_im, [512, 512], division=division).to(device)

                # inference
                result = model(image)

                # post process
                binary_mask = postprocess_image(result[0][0], orig_im_size)
                # threshold the mask
                _, binary_mask = cv2.threshold(binary_mask, 200, 255, cv2.THRESH_BINARY)

                n_pixels = np.prod(binary_mask.shape)
                if np.sum(binary_mask / 255) < 0.15 * n_pixels or np.sum(binary_mask / 255) > 0.6 * n_pixels:
                    division -= 1
                    continue
                break

            # Dilate the mask
            kernel = np.ones((21, 21), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            normal_path = p[-1]
            normal_im = io.imread(normal_path)
            normal_im = cv2.resize(normal_im, (512, 512))
            masked_normal_image = cv2.bitwise_and(normal_im, normal_im, mask=binary_mask)

            # save the masked normal image
            masked_normal_image_path = normal_path.replace("pred_normals.png", "masked_pred_normals.png")
            io.imsave(masked_normal_image_path, masked_normal_image)
