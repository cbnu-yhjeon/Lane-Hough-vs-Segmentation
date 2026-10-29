"""
L2: 경량 다중 작업 분할 모델

인코더  MobileNetV3-Large (ImageNet 사전학습, torchvision)
디코더  U-Net식 skip 연결 (stride 32 → 2), 헤드 출력을 2배 업샘플
헤드    주행가능영역 3-class (배경 / 자차 차로 / 옆 차로) + 차선 1-class

차선은 몇 픽셀 두께라서 stride 16에서 끝나는 DeepLabV3 출력으로는 끊깁니다.
그래서 얕은 층의 skip을 모두 쓰는 U-Net 디코더를 붙였습니다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

# features[i] 출력의 stride: 1→2, 3→4, 6→8, 12→16, 16→32
SKIP_IDX = (1, 3, 6, 12, 16)
SKIP_CH = (16, 24, 40, 112, 960)
DEC_CH = (32, 48, 64, 96, 128)


def conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(conv_bn(cin + cskip, cout), conv_bn(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class LaneSegNet(nn.Module):
    def __init__(self, pretrained: bool = True, n_drivable: int = 3):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.center = conv_bn(SKIP_CH[-1], DEC_CH[-1], 1)
        self.ups = nn.ModuleList(
            UpBlock(DEC_CH[i + 1], SKIP_CH[i], DEC_CH[i]) for i in reversed(range(4)))
        self.head_drivable = nn.Sequential(conv_bn(DEC_CH[0], 32), nn.Conv2d(32, n_drivable, 1))
        self.head_lane = nn.Sequential(conv_bn(DEC_CH[0], 32), nn.Conv2d(32, 1, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B,3,H,W) RGB [0,1]. 반환: drivable (B,3,H,W), lane (B,1,H,W) logits."""
        size = x.shape[-2:]
        h = (x - self.mean) / self.std
        skips = []
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i in SKIP_IDX:
                skips.append(h)
        h = self.center(skips[-1])
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            h = up(h, skip)
        # 헤드는 stride 2에서 계산하고 logit만 원래 크기로 올립니다 (학습 속도 약 1.5배)
        up = lambda t: F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return up(self.head_drivable(h)), up(self.head_lane(h))
