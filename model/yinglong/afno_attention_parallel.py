import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from typing import Tuple


import torch.nn.init as init
from .afnonet import AFNO2D
from .afnonet import Mlp
from .afnonet import PatchEmbed
from .time_embedding import TimeFeatureEmbedding


from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Block(nn.Module):
    """AFNO network block.

    Args:
        dim (int): The input tensor dimension.
        mlp_ratio (float, optional): The ratio used in MLP. Defaults to 4.0.
        drop (float, optional): The drop ratio used in MLP. Defaults to 0.0.
        drop_path (float, optional): The drop ratio used in DropPath. Defaults to 0.0.
        activation (str, optional): Name of activation function. Defaults to "gelu".
        norm_layer (nn.Layer, optional): Class of norm layer. Defaults to nn.LayerNorm.
        double_skip (bool, optional): Whether use double skip. Defaults to True.
        num_blocks (int, optional): The number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
    """

    def __init__(
        self,
        dim: int,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        attn_channel_ratio=0.5,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        activation: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        double_skip: bool = True,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn_channle_ratio = attn_channel_ratio
        self.attn_dim = int(dim * attn_channel_ratio)

        self.norm1 = norm_layer(dim)

        if dim - self.attn_dim > 0:
            self.filter = AFNO2D(
                dim - self.attn_dim,
                num_blocks,
                sparsity_threshold,
                hard_thresholding_fraction,
            )
        if self.attn_dim > 0:
            self.attn = WindowAttention(
                self.attn_dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                H, W = self.input_resolution
                Hp = int(np.ceil(H / self.window_size)) * self.window_size
                Wp = int(np.ceil(W / self.window_size)) * self.window_size
                img_mask = torch.zeros([1, Hp, Wp, 1])  # 1 Hp Wp 1
                h_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                w_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        try:
                            img_mask[:, h, w, :] = cnt
                        except:
                            pass

                        cnt += 1

                mask_windows = window_partition(
                    img_mask, self.window_size
                )  # nW, window_size, window_size, 1
                mask_windows = mask_windows.reshape(
                    [-1, self.window_size * self.window_size]
                )
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                huns = -100.0 * torch.ones_like(attn_mask)
                attn_mask = huns * (attn_mask != 0)
            else:
                attn_mask = None

            self.register_buffer("attn_mask", attn_mask)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=activation,
            drop=drop,
        )
        self.double_skip = double_skip

    def attn_forward(self, x):
        B, H, W, C = x.shape

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, pad_l, 0,pad_r, 0, pad_b, 0, pad_t])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size, C]
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp, C
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]
        return x

    def afno_forward(self, x):
        x = self.filter(x)
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        residual = x
        x = self.norm1(x)

        if self.attn_dim == 0:
            x = self.afno_forward(x)
        elif self.attn_dim == self.dim:
            x = self.attn_forward(x)
        else:  
            x_attn = x[:, :, :, : self.attn_dim]
            x_afno = x[:, :, :, self.attn_dim :]
            x_attn = self.attn_forward(x_attn)
            x_afno = self.afno_forward(x_afno)

            x = torch.concat([x_attn, x_afno], axis=-1)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x




class AFNOAttnParallelNet(nn.Module):
    """Adaptive Fourier Neural Network.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        img_size (Tuple[int, ...], optional): Image size. Defaults to (720, 1440).
        patch_size (Tuple[int, ...], optional): Path. Defaults to (8, 8).
        in_channels (int, optional): The input tensor channels. Defaults to 20.
        out_channels (int, optional): The output tensor channels. Defaults to 20.
        embed_dim (int, optional): The embedding dimension for PatchEmbed. Defaults to 768.
        depth (int, optional): Number of transformer depth. Defaults to 12.
        mlp_ratio (float, optional): Number of ratio used in MLP. Defaults to 4.0.
        drop_rate (float, optional): The drop ratio used in MLP. Defaults to 0.0.
        drop_path_rate (float, optional): The drop ratio used in DropPath. Defaults to 0.0.
        num_blocks (int, optional): Number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
        num_timestamps (int, optional): Number of timestamp. Defaults to 1.

    """

    def __init__(
        self,
        img_size: Tuple[int, ...] = (720, 1440),
        patch_size: Tuple[int, ...] = (8, 8),
        in_channels: int = 20,
        out_channels: int = 20,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        num_timestamps: int = 1,
        window_size=7,
        num_heads=8,
        attn_channel_ratio=0.5,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_timestamps = num_timestamps
        self.window_size = window_size
        self.num_heads = num_heads
        if not isinstance(attn_channel_ratio, list):
            self.attn_channel_ratio = [attn_channel_ratio] * depth
        else:
            self.attn_channel_ratio = attn_channel_ratio
        assert len(self.attn_channel_ratio) == depth

        norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

  
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.time_embed = TimeFeatureEmbedding(d_model=embed_dim, embed_type='fixed', freq='h')
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    input_resolution=(self.h, self.w),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    attn_channel_ratio=self.attn_channel_ratio[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        # self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            self.out_channels * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward_tensor(self, x, x_time):
        B = x.shape[0]
        x = self.patch_embed(x)
     
        x = x + self.pos_embed + self.time_embed(x_time,x.shape[1])
        x = self.pos_drop(x)

        x = x.reshape((B, self.h, self.w, self.embed_dim))

        for block in self.blocks:
            x = block(x)

        x = self.head(x)

        b = x.shape[0]
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        h = self.img_size[0] // self.patch_size[0]
        w = self.img_size[1] // self.patch_size[1]
        c_out = x.shape[3] // (p1 * p2)
        x = x.reshape((b, h, w, p1, p2, c_out))
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape((b, c_out, h * p1, w * p2))

        return x

    def forward(self, x, x_time, geo):
        geo = geo.unsqueeze(1)        
        y = []
        out = x

        for i in range(self.num_timestamps):
            input = torch.cat((out, geo), dim=1)            
            out = self.forward_tensor(input, x_time[:,i])
            y.append(out)
            
        return y
class AFNOAttnParallelNet_marge(nn.Module):
    """Adaptive Fourier Neural Network.

    Args:
        input_keys (Tuple[str, ...]): Name of input keys, such as ("input",).
        output_keys (Tuple[str, ...]): Name of output keys, such as ("output",).
        img_size (Tuple[int, ...], optional): Image size. Defaults to (720, 1440).
        patch_size (Tuple[int, ...], optional): Path. Defaults to (8, 8).
        in_channels (int, optional): The input tensor channels. Defaults to 20.
        out_channels (int, optional): The output tensor channels. Defaults to 20.
        embed_dim (int, optional): The embedding dimension for PatchEmbed. Defaults to 768.
        depth (int, optional): Number of transformer depth. Defaults to 12.
        mlp_ratio (float, optional): Number of ratio used in MLP. Defaults to 4.0.
        drop_rate (float, optional): The drop ratio used in MLP. Defaults to 0.0.
        drop_path_rate (float, optional): The drop ratio used in DropPath. Defaults to 0.0.
        num_blocks (int, optional): Number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
        num_timestamps (int, optional): Number of timestamp. Defaults to 1.

    """

    def __init__(
        self,
        img_size: Tuple[int, ...] = (720, 1440),
        patch_size: Tuple[int, ...] = (8, 8),
        in_channels: int = 20,
        out_channels: int = 20,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        num_timestamps: int = 1,
        window_size=7,
        num_heads=8,
        attn_channel_ratio=0.5,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_timestamps = num_timestamps
        self.window_size = window_size
        self.num_heads = num_heads
        if not isinstance(attn_channel_ratio, list):
            self.attn_channel_ratio = [attn_channel_ratio] * depth
        else:
            self.attn_channel_ratio = attn_channel_ratio
        assert len(self.attn_channel_ratio) == depth

        norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

  
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        self.time_embed = TimeFeatureEmbedding(d_model=embed_dim, embed_type='fixed', freq='h')
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    input_resolution=(self.h, self.w),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    attn_channel_ratio=self.attn_channel_ratio[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        # self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim,
            self.out_channels * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )
        
        self.wm = torch.tensor(np.load('data_stats/ED_stat/wm69.npy'), dtype=torch.float32)
        self.wn = torch.tensor(np.load('data_stats/ED_stat/wn69.npy'), dtype=torch.float32)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward_tensor(self, x, x_time):
        B = x.shape[0]
        x = self.patch_embed(x)
     
        x = x + self.pos_embed + self.time_embed(x_time,x.shape[1])
        x = self.pos_drop(x)

        x = x.reshape((B, self.h, self.w, self.embed_dim))

        for block in self.blocks:
            x = block(x)

        x = self.head(x)

        b = x.shape[0]
        p1 = self.patch_size[0]
        p2 = self.patch_size[1]
        h = self.img_size[0] // self.patch_size[0]
        w = self.img_size[1] // self.patch_size[1]
        c_out = x.shape[3] // (p1 * p2)
        x = x.reshape((b, h, w, p1, p2, c_out))
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape((b, c_out, h * p1, w * p2))

        return x

    def forward(self, x, x_time, geo, marge):
        self.wm = self.wm.to(x.device)
        self.wn = self.wn.to(x.device)
        geo = geo.unsqueeze(1)        
        y = []
        g = []
        out = x
        
        for i in range(self.num_timestamps):
            # input = torch.cat((out, geo), dim=1)           
            # out = self.forward_tensor(input, x_time[:,i])
            out = marge[:,i]
            y.append(out)
            # out = out * self.wm + marge[:,i] * self.wn
        return y