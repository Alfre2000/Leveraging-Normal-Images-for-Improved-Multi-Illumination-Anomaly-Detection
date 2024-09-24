from evaluation_rd4ad import load_model, predict
from dataset import EyecandiesDataset
from constants import CLASSES
from torch.utils.data import DataLoader
import torch
from utils import min_max_norm, cvt2heatmap, show_cam_on_image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize():
    class_name = "first"

    base_path = "./checkpoints/rd4ad/avg/zero_shot/"
    model_rgb = load_model("rd4ad", base_path + "rgb/first.pth", "rgb", grouped=False)
    model_normal = load_model("rd4ad", base_path + "normal/first.pth", "normal", grouped=False)
    model_normal_real = load_model(
        "rd4ad", base_path + "normal_real/first.pth", "normal_real", grouped=False
    )


    test_classes = CLASSES[5:] if class_name == "first" else CLASSES[:5]
    dataset_path = "../eyecandies-dataset/"
    paths = [f"{dataset_path}/{c}" for c in test_classes]

    test_data_rgb = EyecandiesDataset(
        root=paths, phase="test", method="rgb", grouped=False, size=256, isize=256
    )
    test_dataloader_rgb = DataLoader(test_data_rgb, batch_size=1, shuffle=False)
    test_data_normal = EyecandiesDataset(
        root=paths, phase="test", method="normal", grouped=False, size=256, isize=256
    )
    test_dataloader_normal = DataLoader(test_data_normal, batch_size=1, shuffle=False)
    test_data_normal_real = EyecandiesDataset(
        root=paths, phase="test", method="normal_real", grouped=False, size=256, isize=256
    )
    test_dataloader_normal_real = DataLoader(test_data_normal_real, batch_size=1, shuffle=False)


    names = ["RGB", "Rec. Normal", "Normal Real"]
    dataloaders = [test_dataloader_rgb, test_dataloader_normal, test_dataloader_normal_real]
    models = [model_rgb, model_normal, model_normal_real]

    results = {name: [] for name in names}


    for name, dataloader, model in zip(names, dataloaders, models):
        for sample in dataloader:
            img, gt, label, types = sample
            with torch.no_grad():
                am = predict(*model, img, method="average", grouped=False, sigma=4)

            img = cv2.cvtColor(img[0].permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img) * 255)
            gt = gt.cpu().numpy().astype(int)[0][0] * 255

            results[name].append([img, gt, am])

        maps = min_max_norm([row[2] for row in results[name]])
        maps = [cvt2heatmap(255 - map * 255) for map in maps]
        for i in range(len(results[name])):
            results[name][i][2] = maps[i]


    for x in range(25):
        # Visualize
        fig, axs = plt.subplots(5, 12, figsize=(24, 10), constrained_layout=True)

        # Loop through grids and set images
        indexes = [0 + x, 50 + x, 100 + x, 150 + x, 200 + x, 49 - x, 99 - x, 199 - x, 149 - x, 249 - x]
        for i, j in enumerate(indexes):
            input_image = results["RGB"][j][0]
            ground_truth = results["RGB"][j][1]
            axs[0, i + 1].imshow(input_image)
            axs[1, i + 1].imshow(ground_truth)
            axs[2, i + 1].imshow(show_cam_on_image(input_image, results["RGB"][j][2]))
            axs[3, i + 1].imshow(show_cam_on_image(input_image, results["Rec. Normal"][j][2]))
            axs[4, i + 1].imshow(show_cam_on_image(input_image, results["Normal Real"][j][2]))

        # Set column titles
        for ax, col in zip(axs[0][1:], [*test_classes, *test_classes]):
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
        fig.text(0.07, 0.86, 'Input', ha='center', va='center', rotation='vertical', fontsize=14)
        fig.text(0.07, 0.66, 'GT', ha='center', va='center', rotation='vertical', fontsize=14)
        fig.text(0.07, 0.46, 'RGB', ha='center', va='center', rotation='vertical', fontsize=14)
        fig.text(0.07, 0.26, 'Rec. Normal', ha='center', va='center', rotation='vertical', fontsize=14)
        fig.text(0.07, 0.06, 'Normal Real', ha='center', va='center', rotation='vertical', fontsize=14)

        plt.savefig(dpi=200, fname=f"zero_shot_results/results_{x}.png")
