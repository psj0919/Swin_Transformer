import torch
import torch.nn as nn
from model.Patch_partition import PatchPartition
from model.SwinBlock import SwinBlock
from model.Patch_merging import PatchMerging
from model.decoder.uper_head import UPerHead

class SwinTransformer(nn.Module):
    def __init__(self,
                 dim=(96, 192, 384, 768),
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 resolutions=(56, 28, 14, 7),
                 num_classes=1000):
        super().__init__()

        self.dim = dim
        self.decode = UPerHead(in_channels=[192, 384, 768, 1536], channels=512, num_classes=num_classes)
        self.patch_partition = PatchPartition(dim=dim[0])
        self.stage1 = nn.Sequential(*[
            SwinBlock(dim[0], num_heads[0], input_resolution=(resolutions[0], resolutions[0]))
            for _ in range(depths[0] // 2)
        ])
        self.merge1 = PatchMerging(dim[0], dim[1], (resolutions[0], resolutions[0]))

        # Stage 2
        self.stage2 = nn.Sequential(*[
            SwinBlock(dim[1], num_heads[1], input_resolution=(resolutions[1], resolutions[1]))
            for _ in range(depths[1] // 2)
        ])
        self.merge2 = PatchMerging(dim[1], dim[2], (resolutions[1], resolutions[1]))

        # Stage 3
        self.stage3 = nn.Sequential(*[
            SwinBlock(dim[2], num_heads[2], input_resolution=(resolutions[2], resolutions[2]))
            for _ in range(depths[2] // 2)
        ])
        self.merge3 = PatchMerging(dim[2], dim[3], (resolutions[2], resolutions[2]))

        # Stage 4
        self.stage4 = nn.Sequential(*[
            SwinBlock(dim[3], num_heads[3], input_resolution=(resolutions[3], resolutions[3]))
            for _ in range(depths[3] // 2)
        ])

    def forward(self, x):
        # x: [B, 3, 224, 224]
        batch_size = x.shape[0]
        x = self.patch_partition(x)    # -> [B, H/4, W/4, C]

        c1 = self.stage1(x)           # C1: 1/4 resolution
        x = self.merge1(c1)

        c2 = self.stage2(x)           # C2: 1/8 resolution
        x = self.merge2(c2)

        c3 = self.stage3(x)           # C3: 1/16 resolution
        x = self.merge3(c3)

        c4 = self.stage4(x)           # C4: 1/32 resolution

        c1 = c1.permute(0, 2, 1).view(batch_size, self.dim[0], 56, 56)    #[B, 32, 56, 56]
        c2 = c2.permute(0, 2, 1).view(batch_size, self.dim[1], 28, 28)    #[B, 64, 28, 28]
        c3 = c3.permute(0, 2, 1).view(batch_size, self.dim[2], 14, 14)   #[B, 128, 14, 14]
        c4 = c4.permute(0, 2, 1).view(batch_size, self.dim[3], 7, 7)     #[B, 256, 7, 7]

        feats = [c1, c2, c3, c4]

        output = self.decode(feats)

        return output

if __name__=='__main__':
    model = SwinTransformer(dim = (32, 64, 128, 256), depths=(2, 2, 4, 2),
                        num_heads=(3, 6, 12, 24),
                        resolutions=(56, 28, 14, 7),
                        batch_size=1,
                        num_classes=21)

    input_tensor = torch.randn(1, 3, 224, 224)

    out = model(input_tensor)