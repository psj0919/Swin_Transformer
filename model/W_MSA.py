import torch
import torch.nn as nn
import math
# from timm.models.layers import trunc_normal_

def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

class W_MSA(nn.Module):
    def __init__(self,
                 dim, num_heads, head_dim=None, window_size=7,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * 7 - 1) * (2 * 7 - 1), num_heads))
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(7, 7))
        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        # Shape of x: (B, L, dim)
        # If patch size=4 and image size=224, L=(224/4)**2=3136.
        # Note that
        B, L, C = x.shape
        ws = self.window_size
        w = h = int(math.sqrt(L))

        # --- Efficient batch computation ---
        # This window is for efficient batch computation
        h_ = int(h // ws)
        w_ = int(w // ws)
        #
        x = x.view(B, h, w, C)
        x = x.view(B, h_, ws, w_, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B * h_ * w_, ws * ws, C)
        # Shape of x: (B', L', C)
        # -----------------------------------

        # --- Attention ---
        B_, N, C = x.shape
        # C is "dim".
        qkv = self.qkv(x).view(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1 ,4)
        # Shape of qkv: (3, B_, self.num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = q @ k.transpose(-2, -1)
        # Shape of attn: (B_, self.num_heads, N, N)
        attn = self.softmax(attn * self.scale)
        attn = self.attn_drop(attn)
        # Shape of attn: (B_, self.num_heads, N, N)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Shape of attn: (B_, N, head_dim * self.num_heads)
        x = self.proj(x)
        # Shape of proj: (B_, N, dim)
        x = self.proj_drop(x)
        # Shape of proj: (B_, N, dim)
        # -----------------

        # --- Make multi-batch tensor original batch tensor ---
        x = x.view(B, h_, w_, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, h_ * ws, w_ * ws, -1)
        x = x.view(B, h_ * ws * w_ * ws, C)
        # -----------------------------------------------------

        return x