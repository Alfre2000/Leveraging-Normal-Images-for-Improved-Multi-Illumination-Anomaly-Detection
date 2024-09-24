from tqdm import tqdm as ProgressBar
from torch.utils.data import DataLoader
from dataset import EyecandiesDataset
import torch
import numpy as np
from utils import cosine_similarity_loss, setup_seed, count_parameters
import os
import time
from architectures.dinomaly import vit_encoder
from architectures.dinomaly.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from architectures.dinomaly.uad import ViTillCombined, AttentionFusionModule, ViTill, AttentionFusionModuleNext, AttentionFusionModuleNext2
# from studies.multi_modal_fusion import AttentionFusionBlock
from torch import nn
from functools import partial
from evaluation import evaluation
from constants import CLASSES


def train(category, method, grouped, dataset_path, zero_shot):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['XFORMERS_DISABLED'] = "1"
    torch.cuda.empty_cache()
    setup_seed(42)

    EPOCHS = 60
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8

    ckp_dir = f'./checkpoints/dino/{"fusion" if grouped else "avg"}/{"zero_shot/" if zero_shot else ""}{method}'
    os.makedirs(ckp_dir, exist_ok=True)
    ckp_path = f'{ckp_dir}/{category}.pth'

    if zero_shot:
        train_root = [f"{dataset_path}/{c}" for c in CLASSES if c != category]
        test_root = f"{dataset_path}/{category}"
    else:
        train_root = f"{dataset_path}/{category}"
        test_root = f"{dataset_path}/{category}"

    train_data = EyecandiesDataset(root=train_root, phase="train", method=method, grouped=grouped, size=256, isize=392)
    test_data = EyecandiesDataset(root=test_root, phase="test", method=method, grouped=grouped, size=256, isize=392)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    embed_dim, num_heads = 768, 12

    encoder = vit_encoder.load()
    # do not train the encoder
    for param in encoder.parameters():
        param.requires_grad = False

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., attn=LinearAttention2)
        for _ in range(8)
    ])

    params = {
        "encoder": encoder,
        "bottleneck": bottleneck,
        "decoder": decoder,
        "target_layers": target_layers,
        "fuse_layer_encoder": fuse_layer_encoder,
        "fuse_layer_decoder": fuse_layer_decoder,
    }

    if grouped:
        num_images = 6 if method == 'rgb' else 7
        attension_fusion = AttentionFusionModule(num_images)
        model = ViTillCombined(**params, attention_fusion=attension_fusion)
    else:
        model = ViTill(**params)

    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    optimizer = torch.optim.Adam(trainable.parameters(), lr=LEARNING_RATE)

    for m in trainable.modules():
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    n_parameters = count_parameters(model)
    print(f"Total number of trainable parameters: {n_parameters} ({round(n_parameters / 1_000_000, 1)}M)")
    print(f"Encoder parameters: {count_parameters(encoder)}")
    if grouped:
        print(f"Attention Fusion parameters: {count_parameters(attension_fusion)}")
        # print(f"RGB Fusion parameters: {count_parameters(attension_fusion.rgb_attention_fcs)}")
        # print(f"Modal Fusion parameters: {count_parameters(attension_fusion.modal_fusion_blocks)}")
    print(f"Bottleneck parameters: {count_parameters(bottleneck)}")
    print(f"Decoder parameters: {count_parameters(decoder)}")

    max_image_auroc = 0
    max_pixel_avg_precision = 0
    epoch_final_model = 0
    for epoch in range(EPOCHS):
        model.train()
        loss_list = []
        for img in ProgressBar(train_dataloader):
            if grouped:
                img = [i.to(device) for i in img]
            else:
                img = img.to(device)
            inputs, outputs = model(img)
            loss = cosine_similarity_loss(inputs, outputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCHS, np.mean(loss_list)))

        # if the loss is nan, stop training
        # if np.isnan(np.mean(loss_list)):
        #     break

        if (epoch + 1) % 5 == 0:
            maps_combination = "average" if grouped or method == 'rgb' else "same_weights"
            start = time.time()
            metrics = evaluation(model, test_dataloader, grouped=grouped, method=maps_combination, n_tresh=20)
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
                torch.save(model.state_dict(), ckp_path)

    print(f"Best model found at epoch {epoch_final_model} with Sample AUROC: {max_image_auroc}")
    print("Finished training for category:", category)
