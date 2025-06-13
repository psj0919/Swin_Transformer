import torch
import torch.nn as nn
from model.Patch_partition import PatchPartition
from model.SwinBlock import SwinBlock
from model.Patch_merging import PatchMerging


class SwinTransformer(nn.Module):
    def __init__(self,
                 dim=(96, 192, 384, 768),
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 resolutions=(56, 28, 14, 7),
                 num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # patch partition
            PatchPartition(dim=dim[0]),

            # swin block 1
            nn.Sequential(*[SwinBlock(dim[0], num_heads[0], input_resolution=(resolutions[0], resolutions[0])) for _ in
                            range(depths[0] // 2)]),
            # patch merging 1
            PatchMerging(dim[0], dim[1], (resolutions[0], resolutions[0])),

            # swin block 2
            nn.Sequential(*[SwinBlock(dim[1], num_heads[1], input_resolution=(resolutions[1], resolutions[1])) for _ in
                            range(depths[1] // 2)]),
            # patch merging 2
            PatchMerging(dim[1], dim[2], (resolutions[1], resolutions[1])),

            # swin block 3
            nn.Sequential(*[SwinBlock(dim[2], num_heads[2], input_resolution=(resolutions[2], resolutions[2])) for _ in
                            range(depths[2] // 2)]),
            # patch merging 3
            PatchMerging(dim[2], dim[3], (resolutions[2], resolutions[2])),

            # swin block 4
            nn.Sequential(*[SwinBlock(dim[3], num_heads[3], input_resolution=(resolutions[3], resolutions[3])) for _ in
                            range(depths[3] // 2)]),
        )
        self.norm = nn.LayerNorm(dim[3])
        self.head = nn.Linear(dim[3], num_classes)

    def forward(self, x):
        """
        :param x: [B, 3, 224, 224]
        :return:
        """
        x = self.features(x)  # [B, 49, 768]
        x = self.norm(x)      # [B, 49, 768]
        x = x.mean(dim=1)     # [B, 768]
        x = self.head(x)      # [B, 1000]
        return x