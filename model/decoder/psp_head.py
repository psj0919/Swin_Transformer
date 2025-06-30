import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    """
    Pooling Pyramid Module (PPM) for PSPNet.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners=False):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners

        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=False)
                )
            )

    def forward(self, x):
        """Forward function for PPM."""
        outs = []
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(
                out,
                size=x.size()[2:],  # same spatial size as input
                mode='bilinear',
                align_corners=self.align_corners
            )
            outs.append(out)
        return outs
