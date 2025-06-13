import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, downscaling_factor=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.norm = nn.LayerNorm(in_channels * downscaling_factor ** 2)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels, bias=False)

    def forward(self, x):
        b, l, c = x.shape
        h, w = self.input_resolution
        x = x.view(b, h, w, c)
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = x.view(-1, new_h * new_w, c * self.downscaling_factor ** 2)
        x = self.norm(x)
        x = self.linear(x)
        return x