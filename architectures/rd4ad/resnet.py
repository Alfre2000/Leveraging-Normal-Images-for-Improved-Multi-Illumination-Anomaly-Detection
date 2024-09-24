import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Optional
from torch.nn import functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class RD4ADFusion(nn.Module):
    def __init__(self, encoder, bottleneck, decoder, attention_fusion):
        super(RD4ADFusion, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.attention_fusion = attention_fusion

    def forward(self, images):
        features = []
        for i in range(len(images)):
            features.append(self.encoder(images[i]))

        inputs = self.attention_fusion(features)
        outputs = self.decoder(self.bottleneck(inputs))
        return inputs, outputs


class RD4AD(nn.Module):
    def __init__(self, encoder, bottleneck, decoder):
        super(RD4AD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, image):
        inputs = self.encoder(image)
        outputs = self.decoder(self.bottleneck(inputs))
        return inputs, outputs


class AttentionFusionModule(nn.Module):
    def __init__(self, num_images):
        super(AttentionFusionModule, self).__init__()
        self.num_images = num_images
        self.num_layers = 3
        self.embed_dims = [256, 512, 1024]
        self.feature_dims = [64, 32, 16]
        self.attention_fcs = nn.ModuleList([
            nn.Linear(256 * 64 * 64 * self.num_images, self.num_images),
            nn.Linear(512 * 32 * 32 * self.num_images, self.num_images),
            nn.Linear(1024 * 16 * 16 * self.num_images, self.num_images)
        ])

    def forward(self, feature_sets):
        batch_size = feature_sets[0][0].size()[0]
        final_fused_features = []

        for i in range(self.num_layers):
            features = torch.stack([feature_sets[j][i].view(batch_size, -1) for j in range(self.num_images)], dim=1)
            flattened_features = features.view(batch_size, -1)
            attention_weights = F.softmax(self.attention_fcs[i](flattened_features), dim=1)
            fused_features = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)
            reshaped_features = fused_features.view(batch_size, self.embed_dims[i], self.feature_dims[i], self.feature_dims[i])
            final_fused_features.append(reshaped_features)

        return final_fused_features


class WideResNet(nn.Module):
    pretrained_url = "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"

    def __init__(self) -> None:
        super(WideResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.num_classes = 1000
        replace_stride_with_dilation = [False, False, False]
        self.base_width = 64 * 2
        self.layers = [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, self.layers[0])
        self.layer2 = self._make_layer(128, self.layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(256, self.layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(512, self.layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load pretrained model
        state_dict = load_state_dict_from_url(self.pretrained_url, progress=True)
        self.load_state_dict(state_dict)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                norm_layer(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(
            self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(self.inplanes, planes, groups=self.groups,
                           base_width=self.base_width, dilation=self.dilation,
                           norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature_a = self.layer1(x)
        feature_b = self.layer2(feature_a)
        feature_c = self.layer3(feature_b)
        _ = self.layer4(feature_c)


        return [feature_a, feature_b, feature_c]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttnBottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        attention: bool = True,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        self.attention = attention
        #  print("Attention:",self.attention)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        #  self.cbam = GLEAM([int(planes * self.expansion/4),
        #                   int(planes * self.expansion//2),
        #                   planes * self.expansion], 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        #  if self.attention:
        #    x = self.cbam(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)

        return out


class BN_layer(nn.Module):
    width_per_group = 64 * 2

    def __init__(self,
                 layers: int,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
        super(BN_layer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = self.width_per_group
        self.inplanes = 256 * AttnBottleneck.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(512, layers, stride=2)

        self.conv1 = conv3x3(64 * AttnBottleneck.expansion, 128 * AttnBottleneck.expansion, 2)
        self.bn1 = norm_layer(128 * AttnBottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * AttnBottleneck.expansion, 256 * AttnBottleneck.expansion, 2)
        self.bn2 = norm_layer(256 * AttnBottleneck.expansion)
        self.conv3 = conv3x3(128 * AttnBottleneck.expansion, 256 * AttnBottleneck.expansion, 2)
        self.bn3 = norm_layer(256 * AttnBottleneck.expansion)

        self.conv4 = conv1x1(1024 * AttnBottleneck.expansion, 512 * AttnBottleneck.expansion, 1)
        self.bn4 = norm_layer(512 * AttnBottleneck.expansion)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * AttnBottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * AttnBottleneck.expansion, stride),
                norm_layer(planes * AttnBottleneck.expansion),
            )

        layers = []
        layers.append(AttnBottleneck(self.inplanes * 3, planes, stride, downsample, self.groups,
                      self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * AttnBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(AttnBottleneck(self.inplanes, planes, groups=self.groups,
                          base_width=self.base_width, dilation=self.dilation,
                          norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        #  x = self.cbam(x)
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        output = self.bn_layer(feature)
        #  x = self.avgpool(feature_d)
        #  x = torch.flatten(x, 1)
        #  x = self.fc(x)

        return output.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
