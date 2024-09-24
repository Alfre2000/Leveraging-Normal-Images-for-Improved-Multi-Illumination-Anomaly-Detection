import torch
from ad_types import Method
from architectures.dinomaly import vit_encoder
from architectures.dinomaly.uad import ViTill, AttentionFusionModule, ViTillCombined
from functools import partial
from architectures.dinomaly.vision_transformer import bMlp, LinearAttention2, Block as VitBlock
import torch.nn as nn


def load_model(model_path, method: Method, grouped: bool) -> torch.nn.Module:
    """
    Load the model from the given model path.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = vit_encoder.load()

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    embed_dim, num_heads = 768, 12

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    if not grouped:
        model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                       mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder,
                       fuse_layer_decoder=fuse_layer_decoder)
    else:
        num_images = 6 if method == 'rgb' else 7
        attension_fusion = AttentionFusionModule(num_images)
        model = ViTillCombined(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                               fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder,
                               attention_fusion=attension_fusion)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
