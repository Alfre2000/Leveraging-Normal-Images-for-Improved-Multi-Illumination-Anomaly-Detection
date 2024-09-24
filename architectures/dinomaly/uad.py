import torch
from torch import nn
import math
from torch.nn import functional as F
from functools import reduce
from torch.nn.modules.utils import _pair
from operator import mul


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all



class ViTillCombined(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            attention_fusion,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
    ) -> None:
        super(ViTillCombined, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.attention_fusion = attention_fusion
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, images):
        all_en_lists = []
        for img in images:
            x = self.encoder.prepare_tokens(img)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
                if i in self.target_layers:
                    en_list.append(x)
            all_en_lists.append(en_list)

        en_list = self.attention_fusion(all_en_lists)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=None)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
        de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class AttentionFusionModule(nn.Module):
    def __init__(self, num_images: int, embed_dim: int = 789):
        super(AttentionFusionModule, self).__init__()
        self.num_images = num_images
        self.num_layers = 8
        self.embed_dim = embed_dim
        self.feature_dim = 768
        self.attention_fcs = nn.ModuleList([
            nn.Linear(self.embed_dim * self.feature_dim * self.num_images, self.num_images) for _ in range(8)
        ])

    def forward(self, feature_sets):
        batch_size = feature_sets[0][0].size()[0]
        final_fused_features = []

        for i in range(self.num_layers):
            features = torch.stack([feature_sets[j][i].view(batch_size, -1) for j in range(self.num_images)], dim=1)
            flattened_features = features.view(batch_size, -1)
            attention_weights = F.softmax(self.attention_fcs[i](flattened_features), dim=1)
            fused_features = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)
            reshaped_features = fused_features.view(batch_size, self.embed_dim, self.feature_dim)
            final_fused_features.append(reshaped_features)

        return final_fused_features


class AttentionFusionModuleNext(nn.Module):
    def __init__(self, num_images: int = 7, embed_dim: int = 789):
        super(AttentionFusionModuleNext, self).__init__()
        self.num_images = num_images
        self.num_layers = 8
        self.embed_dim = embed_dim
        self.feature_dim = 768
        self.num_rgb_images = 6

        # Attention layers for the 6 RGB images
        self.rgb_attention_fcs = nn.ModuleList([
            nn.Linear(self.embed_dim * self.feature_dim * self.num_rgb_images, self.num_rgb_images) for _ in range(self.num_layers)
        ])

        # Attention layers for fusing the 6 fused images with the normal image
        self.final_attention_fcs = nn.ModuleList([
            nn.Linear(self.embed_dim * self.feature_dim * 2, 2) for _ in range(self.num_layers)
        ])

    def forward(self, feature_sets):
        batch_size = feature_sets[0][0].size()[0]
        final_fused_features = []

        for i in range(self.num_layers):
            # Fuse the 6 RGB images
            rgb_features = torch.stack([feature_sets[j][i].view(batch_size, -1) for j in range(self.num_rgb_images)], dim=1)
            flattened_rgb_features = rgb_features.view(batch_size, -1)
            rgb_attention_weights = F.softmax(self.rgb_attention_fcs[i](flattened_rgb_features), dim=1)
            fused_rgb_features = torch.sum(rgb_features * rgb_attention_weights.unsqueeze(-1), dim=1)
            reshaped_rgb_features = fused_rgb_features.view(batch_size, self.embed_dim, self.feature_dim)

            # Fuse the fused RGB images with the normal image
            normal_image_features = feature_sets[self.num_rgb_images][i].view(batch_size, -1)
            combined_features = torch.cat((reshaped_rgb_features.view(batch_size, -1), normal_image_features), dim=1)
            final_attention_weights = F.softmax(self.final_attention_fcs[i](combined_features), dim=1)
            final_fused_feature = torch.sum(torch.stack((reshaped_rgb_features.view(batch_size, -1), normal_image_features), dim=1) * final_attention_weights.unsqueeze(-1), dim=1)
            reshaped_final_feature = final_fused_feature.view(batch_size, self.embed_dim, self.feature_dim)

            final_fused_features.append(reshaped_final_feature)

        return final_fused_features



class AttentionFusionModuleNext2(nn.Module):
    def __init__(self, num_images: int = 7, embed_dim: int = 789):
        super(AttentionFusionModuleNext2, self).__init__()
        self.num_images = num_images
        self.num_layers = 8
        self.embed_dim = embed_dim
        self.feature_dim = 768
        self.num_rgb_images = 6
        self.reduced_dim = 8  # Reduced dimension

        # Projection layers for each image to reduce dimensionality
        self.rgb_projections = nn.ModuleList([
            nn.Linear(self.embed_dim * self.feature_dim, self.reduced_dim) for _ in range(self.num_rgb_images)
        ])
        self.normal_projection = nn.Linear(self.embed_dim * self.feature_dim, self.reduced_dim)

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Linear(self.reduced_dim * (self.num_rgb_images + 1), self.embed_dim * self.feature_dim) for _ in range(self.num_layers)
        ])

        # Layer Normalization
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(self.embed_dim * self.feature_dim) for _ in range(self.num_layers)
        ])

    def forward(self, feature_sets):
        batch_size = feature_sets[0][0].size()[0]
        final_fused_features = []

        for i in range(self.num_layers):
            # Project each RGB image feature to a reduced dimension
            projected_rgb_features = [self.rgb_projections[j](feature_sets[j][i].view(batch_size, -1)) for j in range(self.num_rgb_images)]
            # Project the normal image feature to a reduced dimension
            projected_normal_feature = self.normal_projection(feature_sets[self.num_rgb_images][i].view(batch_size, -1))
            
            # Concatenate the reduced dimension features
            combined_features = torch.cat(projected_rgb_features + [projected_normal_feature], dim=1)
            
            # Apply fusion layer
            fused_features = self.fusion_layers[i](combined_features)
            
            # Apply Layer Normalization
            normalized_output = self.layer_norm[i](fused_features)
            reshaped_final_feature = normalized_output.view(batch_size, self.embed_dim, self.feature_dim)
            
            final_fused_features.append(reshaped_final_feature)

        return final_fused_features


class PromptVitill(ViTillCombined):
    def __init__(self, num_prompts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_prompts = num_prompts
        self.patch_size = _pair(14)
        self.prompt_dim = 768

        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))

        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.n_prompts, self.prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        self.prompt_proj = nn.Identity()

    def forward(self, images):
        B = images[0].shape[0]
        all_en_lists = []
        for img in images:
            x = self.encoder.prepare_tokens(img)
            x = torch.cat((
                x[:, :1, :],
                self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1),
                x[:, 1:, :]
            ), dim=1)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
                if i in self.target_layers:
                    en_list.append(x)
            all_en_lists.append(en_list)

        en_list = self.attention_fusion(all_en_lists)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens - self.n_prompts))

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=None)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = [e[:, 1 + self.encoder.num_register_tokens + self.n_prompts:, :] for e in en]
        de = [d[:, 1 + self.encoder.num_register_tokens + self.n_prompts:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de
