import os
import yaml
import torch
from PIL import Image
from torchvision import transforms
from ad_types import Method, Phase
from typing import List, Union


def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()
    ])
    return data_transforms, gt_transforms


class EyecandiesDataset(torch.utils.data.Dataset):
    """
    A custom dataset loader for the Eyecandies dataset.

    Parameters:
    - roots (Union[str, List[str]]): Path or list of paths to the dataset directories.
    - phase (Phases): The phase of the dataset to load ('train' or 'test_public').
    - method (Methods): Method to determine which types of images to load, such as RGB, normals, etc.
    - grouped (bool, optional): If True, images will be grouped together based on their sequence. Defaults to False.
    - size (int, optional): The size to which the images will be resized. Defaults to 256.
    - isize (int, optional): Another dimension for image resizing, used in transformations. Defaults to 392.
    """

    RGB_PATHS = ["image_0.png", "image_1.png", "image_2.png", "image_3.png", "image_4.png", "image_5.png"]

    def __init__(
        self,
        root: Union[str, List[str]],
        phase: Phase,
        method: Method,
        grouped: bool = False,
        size: int = 256,
        isize: int = 256
    ):
        assert phase in ['train', 'test']
        assert method in ['rgb', 'rgb_normal', 'normal', 'masked_normal', 'all', 'rgb_normal_real', 'normal_real']

        roots = [root] if isinstance(root, str) else root
        if phase == 'train':
            self.base_paths = [os.path.join(root, 'train', 'data') for root in roots]
        else:
            self.base_paths = [os.path.join(root, 'test_public', 'data') for root in roots]
        self.transform, self.gt_transform = get_data_transforms(size, isize)
        self.phase = phase
        self.roots = roots
        self.grouped = grouped
        self.method = method

        if self.method == 'rgb':
            self.IMG_NAMES = self.RGB_PATHS
        elif self.method == 'rgb_normal':
            self.IMG_NAMES = self.RGB_PATHS + ['pred_normals.png']
        elif self.method == 'normal':
            self.IMG_NAMES = ['pred_normals.png']
        elif self.method == 'normal_real':
            self.IMG_NAMES = ['normals.png']
        elif self.method == 'masked_normal':
            self.IMG_NAMES = ['masked_pred_normals.png']
        elif self.method == 'all':
            self.IMG_NAMES = self.RGB_PATHS + ['pred_normals.png', 'normals.png']
        elif self.method == 'rgb_normal_real':
            self.IMG_NAMES = self.RGB_PATHS + ['normals.png']

        # load dataset
        if phase == 'train':
            self.imgs_paths = self.load_dataset_train()
        else:
            self.imgs_paths, self.gt_data, self.labels, self.types = self.load_dataset_test()

        print(f"Number of {phase} samples: {len(self.imgs_paths)}")

    def load_dataset_train(self):
        data = []
        for i, base_path in enumerate(self.base_paths):
            for x in range(1000):
                n = str(x).rjust(3, '0')
                for name in self.IMG_NAMES:
                    data.append(os.path.join(base_path, f"{n}_{name}"))
            for x in range(100):
                n = str(x).rjust(2, '0')
                for name in self.IMG_NAMES:
                    data.append(os.path.join(self.roots[i], 'val', 'data', f"{n}_{name}"))

            if self.grouped:
                grouped_data = []
                for i in range(0, len(data), len(self.IMG_NAMES)):
                    grouped_data.append(data[i:i + len(self.IMG_NAMES)])
                return grouped_data

        return data

    def load_dataset_test(self):
        anomaly_types = ["anomalous_bumps", "anomalous_dents", "anomalous_colors", "anomalous_normals"]
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        for base_path in self.base_paths:
            for x in range(50):
                n = str(x).rjust(2, '0')

                metadata_path = os.path.join(base_path, f"{n}_metadata.yaml")
                with open(metadata_path, 'r') as file:
                    metadata = yaml.safe_load(file)

                img_tot_paths.append([os.path.join(base_path, f"{n}_{name}") for name in self.IMG_NAMES])
                gt_tot_paths.append(os.path.join(base_path, f"{n}_mask.png"))
                tot_labels.append(metadata['anomalous'])

                anomalies = [anomaly for anomaly in anomaly_types if metadata[anomaly]]
                tot_types.append(anomalies if len(anomalies) > 0 else [])

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        if self.phase == "train":
            if self.grouped:
                imgs = []
                for path in self.imgs_paths[idx]:
                    img_name = os.path.join(path)
                    img = Image.open(img_name)
                    img = img.convert('RGB')
                    img = self.transform(img)
                    imgs.append(img)
                # return imgs, self.imgs_paths[idx]
                return imgs
            img = Image.open(self.imgs_paths[idx])
            img = img.convert('RGB')
            img = self.transform(img)
            return img

        imgs = []
        for path in self.imgs_paths[idx]:
            img = Image.open(path)
            img = img.convert('RGB')
            img = self.transform(img)
            imgs.append(img)

        gt, label, _type = self.gt_data[idx], self.labels[idx], self.types[idx]
        gt = Image.open(gt)
        gt = self.gt_transform(gt)

        # every value > 0.5 should be set to 1
        gt = (gt > 0.5).int()

        # return imgs, self.imgs_paths[idx], gt, label, _type
        return imgs, gt, label, _type
