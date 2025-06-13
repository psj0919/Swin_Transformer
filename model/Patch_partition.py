import torch.nn as nn


class PatchPartition(nn.Module):
    def __init__(self,
                 dim: int = 96,
                 patch_size: int = 4,
                 ):
        """
        this patch partition + Linear Embedding
        :param patch_size:
        """
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Assume: image size [B, 3, 224, 224] with patch size = 4
        x = self.proj(x)                  # [B, 96, 56, 56]
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC, so C=56^2
        x = self.norm(x)
        return x