from evaluation_rd4ad import evaluation
from tqdm import tqdm as ProgressBar
from torch.utils.data import DataLoader
from dataset import EyecandiesDataset
import torch
from utils import cosine_similarity_loss, setup_seed, count_parameters
import os
from architectures.rd4ad.de_resnet import de_wide_resnet50_2
from architectures.rd4ad.resnet import WideResNet, BN_layer, AttentionFusionModule
import time
from constants import CLASSES


def train(category, method, grouped, dataset_path, zero_shot=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['XFORMERS_DISABLED'] = "1"
    torch.cuda.empty_cache()
    setup_seed(42)

    EPOCHS = 60
    LEARNING_RATE = 0.003
    BATCH_SIZE = 8

    ckp_dir = f'./checkpoints/rd4ad/{"fusion" if grouped else "avg"}/{"zero_shot/" if zero_shot else ""}{method}'
    os.makedirs(ckp_dir, exist_ok=True)
    ckp_path = f'{ckp_dir}/{category}.pth'

    if zero_shot:
        train_root = [f"{dataset_path}/{c}" for c in CLASSES[5:]]
        test_root = [f"{dataset_path}/{c}" for c in CLASSES[:5]]
    else:
        train_root = f"{dataset_path}/{category}"
        test_root = f"{dataset_path}/{category}"
    train_data = EyecandiesDataset(root=train_root, phase="train", method=method, grouped=grouped, size=256, isize=256)
    test_data = EyecandiesDataset(root=test_root, phase="test", method=method, grouped=grouped, size=256, isize=256)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # encoder, bn = WideResNet(), BN_layer(3)
    # encoder = encoder.to(device)
    # bn = bn.to(device)
    # encoder.eval()
    # decoder = de_wide_resnet50_2(pretrained=False)
    # decoder = decoder.to(device)
    # num_images = 6 if method == 'rgb' else 7
    # attention = AttentionFusionModule(num_images)
    # attention = attention.to(device)

    # optimizer = torch.optim.Adam(list(
    #     decoder.parameters()) + list(bn.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999)
    # )
    encoder, bn = WideResNet(), BN_layer(3)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    if grouped:
        num_images = 6 if method == 'rgb' else 7
        attention = AttentionFusionModule(num_images)
        print("Using device:", device)
        attention = attention.to(device)

    params = list(decoder.parameters()) + list(bn.parameters())
    if grouped:
        params += list(attention.parameters())

    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    n_params_decoder = count_parameters(decoder)
    n_params_bn = count_parameters(bn)
    # n_params_attention = count_parameters(attention)

    print(f"Number of parameters in the decoder: {n_params_decoder / 1e6} M")
    print(f"Number of parameters in the BN layer: {n_params_bn / 1e6} M")
    # print(f"Number of parameters in the attention module: {n_params_attention / 1e6} M")
    # print(f"Total number of parameters: {(n_params_decoder + n_params_bn + n_params_attention) / 1e6} M")

    max_image_auroc = 0
    max_pixel_avg_precision = 0
    epoch_final_model = 0
    for epoch in range(EPOCHS):
        bn.train()
        decoder.train()
        if grouped:
            attention.train()
        loss_list = []
        for img in ProgressBar(train_dataloader):
            if grouped:
                img = [i.to(device) for i in img]
                inputs = [encoder(i) for i in img]
                inputs = attention(inputs)
            else:
                img = img.to(device)
                inputs = encoder(img)
            outputs = decoder(bn(inputs))
            loss = cosine_similarity_loss(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())


        if (epoch + 1) % 5 == 0:
            maps_combination = "average" if grouped or method == 'rgb' else "same_weights"
            start = time.time()
            metrics = evaluation(encoder, bn, decoder, test_dataloader, grouped=grouped, method=maps_combination, n_tresh=20)
            end = time.time()
            print(", ".join([f"{k}: {v}" for k, v in metrics.items()]))
            print(f"Time taken for evaluation: {end - start:.2f} seconds")

            bigger_sample_auroc = metrics['Sample AUROC'] > max_image_auroc
            same_sample_auroc = metrics['Sample AUROC'] == max_image_auroc
            bigger_pixel_avg_precision = metrics['Pixel Average Precision'] > max_pixel_avg_precision
            if bigger_sample_auroc or (same_sample_auroc and bigger_pixel_avg_precision):
                print(f"Sample AUROC improved from {max_image_auroc} to {metrics['Sample AUROC']}. Saving model...")
                max_image_auroc = metrics['Sample AUROC']
                max_pixel_avg_precision = metrics['Pixel Average Precision']
                epoch_final_model = epoch + 1
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)

    print(f"Best model found at epoch {epoch_final_model} with Sample AUROC: {max_image_auroc}")
    print("Finished training for category:", category)
