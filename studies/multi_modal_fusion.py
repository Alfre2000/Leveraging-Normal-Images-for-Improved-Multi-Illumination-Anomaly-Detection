import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiModalFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7, pool_size=2):
        super(MultiModalFusion, self).__init__()
        # max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.conv_reduce = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm((out_channels, 768 // pool_size))
        self.upsample = nn.Upsample(scale_factor=pool_size, mode='nearest')

    def forward(self, rgb, normal):
        fused = torch.cat([rgb, normal], dim=1)
        fused = self.pool(fused)
        ca = self.channel_attention(fused)
        ca_out = fused * ca
        sa = self.spatial_attention(ca_out)
        sa_out = ca_out * sa
        reduced_out = self.conv_reduce(sa_out)
        normalized = self.norm(reduced_out)
        return self.upsample(normalized)


class AttentionFusionBlock(nn.Module):
    def __init__(self, num_images: int = 7, embed_dim: int = 789):
        super(AttentionFusionBlock, self).__init__()
        self.num_images = num_images
        self.num_layers = 8
        self.embed_dim = embed_dim
        self.feature_dim = 768
        self.num_rgb_images = 6
        self.pool_size = 4

        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=self.pool_size, stride=self.pool_size)

        # Attention layers for the 6 RGB images
        self.rgb_attention_fcs = nn.ModuleList([
            nn.Linear(self.embed_dim * (self.feature_dim // self.pool_size) * self.num_rgb_images, self.num_rgb_images) for _ in range(self.num_layers)
        ])

        # Upsampling layer.
        self.upsample = nn.Upsample(scale_factor=self.pool_size, mode='nearest')

        # Multi Modality Fusion blocks
        self.modal_fusion_blocks = nn.ModuleList([
            MultiModalFusion(
                in_channels=self.embed_dim * 2,
                out_channels=self.embed_dim,
                reduction=16,
                kernel_size=5,
                pool_size=4
            ) for _ in range(self.num_layers)
        ])


    def forward(self, feature_sets):
        batch_size = feature_sets[0][0].size()[0]
        final_fused_features = []

        for i in range(self.num_layers):
            # Max pool the RGB features
            pooled_rgb_features = [self.pool(feature_sets[j][i].view(batch_size, -1)) for j in range(self.num_rgb_images)]
            # Flatten the pooled RGB features
            rgb_features = torch.stack([feat for feat in pooled_rgb_features], dim=1)
            flattened_rgb_features = rgb_features.view(batch_size, -1)

            # Attention weights
            attention_weights = F.softmax(self.rgb_attention_fcs[i](flattened_rgb_features), dim=1)

            # Fuse the RGB features
            fused_rgb_features = torch.sum(rgb_features * attention_weights.unsqueeze(-1), dim=1)
            reshaped_rgb_features = fused_rgb_features.view(batch_size, self.embed_dim, self.feature_dim // self.pool_size)

            # Upsample back to original size
            upsampled_rgb_features = self.upsample(reshaped_rgb_features)

            # Fuse the RGB and Normal features
            normal_image_features = feature_sets[self.num_rgb_images][i]
            fused_features = self.modal_fusion_blocks[i](upsampled_rgb_features, normal_image_features)
            fused_features = fused_features + upsampled_rgb_features + normal_image_features
            final_fused_features.append(fused_features)

        return final_fused_features
