import torch
import torch.nn as nn
from model.W_MSA import W_MSA
from model.MLP import MLP
from model.SW_MSA import SW_MSA

class SwinBlock(nn.Module):

    def __init__(self,
                 dim: int = 96,
                 num_heads: int = 3,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 input_resolution: tuple = (56, 56)):
        super().__init__()

        # for w-msa
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.w_msa = W_MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.mlp1 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # for sw-msa
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        self.sw_msa = SW_MSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, input_resolution=input_resolution)
        self.mlp2 = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.w_msa(self.norm1_1(x))             # [B, 3136, 96]
        x = x + self.mlp1(self.norm1_2(x))              # [B, 3136, 96]

        x = x + self.sw_msa(self.norm2_1(x))            # [B, 3136, 96]
        x = x + self.mlp2(self.norm2_2(x))              # [B, 3136, 96]
        return x