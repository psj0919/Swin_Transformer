import torch
import torch.nn as nn
import torch.nn.functional as F
from model.decoder.wrappers import resize
from model.decoder.psp_head import PPM


class UPerHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, pool_scales=(1, 2, 3, 6), align_corners=False):
        super(UPerHead, self).__init__()

        self.in_channels = in_channels  # list of 4 backbone feature dims (C1, C2, C3, C4)
        self.channels = channels
        self.align_corners = align_corners

        # PSP Module
        self.psp_modules = PPM(pool_scales, self.in_channels[-1], self.channels, align_corners=align_corners)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=False)
        )

        # FPN lateral and output convs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in self.in_channels[:-1]:  # skip top layer
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, self.channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=False)
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=False)
                )
            )

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(self.in_channels) * self.channels, self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=False)
        )

        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        """
        Args:
            inputs: list of 4 feature maps from the backbone (low ? high resolution)
        Returns:
            segmentation logits: [B, num_classes, H, W]
        """
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # psp on top layer (C4)
        psp_out = [inputs[-1]]
        psp_out.extend(self.psp_modules(inputs[-1]))
        psp_out = torch.cat(psp_out, dim=1)
        laterals.append(self.bottleneck(psp_out))

        # top-down FPN
        for i in range(len(laterals) - 1, 0, -1):
            upsample = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=self.align_corners)
            laterals[i - 1] = laterals[i - 1] + upsample

        # apply fpn convs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(self.in_channels) - 1)
        ]
        fpn_outs.append(laterals[-1])  # add top PSP feature

        # unify size
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        output = resize(
            output,
            size=(224, 224),
            mode = 'bilinear',
            align_corners=self.align_corners
        )
        return output



if __name__=='__main__':
    B = 2
    in_channels = [96, 192, 384, 768]
    channels = 512
    num_classes = 21

    feat_shapes = [(56, 56), (28, 28), (14, 14), (7, 7)]

    feats = [
        torch.randn(B, c, h, w)
        for c, (h, w) in zip(in_channels, feat_shapes)
    ]

    model = UPerHead(in_channels=in_channels, channels=channels, num_classes=num_classes)

    out = model(feats)