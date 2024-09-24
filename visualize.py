from utils import cvt2heatmap, min_max_norm, show_cam_on_image
import matplotlib.pyplot as plt
from dataset import EyecandiesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm as ProgressBar
import torch
import cv2
import numpy as np
from constants import CLASSES
from evaluation import predict as predict_dino, load_model
from evaluation_rd4ad import predict as predict_rd4ad
from collections import defaultdict
import os
from ad_types import Method
import pandas as pd
from sklearn.metrics import roc_auc_score

plt.rcParams['font.family'] = 'serif'


def visualize(input_images: list, ground_truths: list, anomaly_maps: list, filename: str = 'results.png'):
    fig, axs = plt.subplots(3, 12, figsize=(24, 6), constrained_layout=True)

    classes = ['CandyCane', 'ChocolateCookie', 'ChocolatePraline',
               'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich',
               'Lollipop', 'Marshmallow', 'PeppermintCandy']
    # classes = ["CandyCane", "ChocolatePraline", "ChocolateCookie", "Confetto"]
    # classes = ["Marshmallow"]

    # Loop through grids and set images
    for j in range(1, len(classes) + 1):
        axs[0, j].imshow(input_images[j - 1])
        axs[1, j].imshow(ground_truths[j - 1])
        axs[2, j].imshow(show_cam_on_image(input_images[j - 1], anomaly_maps[j - 1]))

    # Set column titles
    for ax, col in zip(axs[0][1:], classes):
        ax.set_title(col, fontsize=14)

    # Remove axes
    for ax in axs.flat:
        ax.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.009, 0.01, 0.955])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Anomaly Score', rotation=270, labelpad=20, fontsize=14)
    cbar.set_ticks([])
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)

    # Add row titles
    fig.text(0.07, 0.8, 'Input', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.07, 0.48, 'GT', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.07, 0.15, 'RD4AD', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.savefig(dpi=300, fname=filename)


def visualize_all(
    architecture: str,
    method: Method,
    dataset_path: str,
    grouped: bool = False,
    size: int = 256,
    isize: int = 392,
):
    if architecture == "rd4ad":
        isize = 256

    maps_combination = "same_weights" if method == "rgb_normal" else "average"
    results = defaultdict(lambda: [])
    data = []
    for class_name in CLASSES:
        ckp_path = f'./checkpoints/{architecture}/{"fusion" if grouped else "avg"}/{method}/{class_name}.pth'
        model = load_model(architecture, ckp_path, method, grouped)
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

        for imgs, gt, label, types in ProgressBar(test_dataloader):
            anomalies = ["anomalous_bumps", "anomalous_dents", "anomalous_normals"]
            print("Processing sample...")

            with torch.no_grad():
                if architecture == "dino":
                    anomaly_map = predict_dino(model, imgs, method=maps_combination, grouped=grouped, sigma=4)
                else:
                    anomaly_map = predict_rd4ad(*model, imgs, method=maps_combination, grouped=grouped, sigma=4)

            if architecture == "dino":
                anomaly_map = anomaly_map[68:324, 68:324]
                ground_truth = gt[:, :, 68:324, 68:324]
            else:
                ground_truth = gt

            gt_pixels = (ground_truth > 0.5).int().cpu().numpy().flatten()
            am_pixels = anomaly_map.flatten()
            auroc_px = round(roc_auc_score(gt_pixels, am_pixels), 3) if label == 1 else None

            gt_image = (ground_truth > 0.5).int().max().item()
            am_image = anomaly_map.max().item()

            # the anomaly map is a 392x392 tensor, we need to get the center 256x256
            if architecture == "dino":
                img = imgs[0][:, :, 68:324, 68:324]
                gt = gt.cpu().numpy().astype(int)[0][0][68:324, 68:324] * 255
            else:
                img = imgs[0]
                gt = gt.cpu().numpy().astype(int)[0][0] * 255

            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img) * 255)

            results[class_name].append([img, gt, anomaly_map])

            data.append({
                "Category": class_name,
                "AUROC Pixel": auroc_px,
                "Image Dif": round(abs(gt_image - am_image), 3),
                "anomalous": gt_image == 1,
                "structural": gt_image == 1 and all(t[0] in anomalies for t in types)
            })

        maps = min_max_norm([row[2] for row in results[class_name]])
        maps = [cvt2heatmap(255 - map * 255) for map in maps]
        for i in range(len(results[class_name])):
            results[class_name][i][2] = maps[i]

    print('Visualizing results...')
    for class_name in CLASSES:
        print(f'Visualizing results for {class_name}: {len(results[class_name])} samples...')


    visualize_dir = f'./visualize/{architecture}/{"fusion" if grouped else "avg"}/{method}/'
    os.makedirs(visualize_dir, exist_ok=True)

    print([len(results[class_name]) for class_name in CLASSES])
    for n in range(len(results[class_name])):
        inputs = [results[class_name][n][0]for class_name in CLASSES]
        gts = [results[class_name][n][1]for class_name in CLASSES]
        maps = [results[class_name][n][2]for class_name in CLASSES]
        visualize(
            input_images=inputs,
            ground_truths=gts,
            anomaly_maps=maps,
            filename=f'{visualize_dir}result_{n}.png'
        )

    df = pd.DataFrame(data)
    df["n"] = df.index
    df.to_csv(f'{visualize_dir}results.csv', index=False)
