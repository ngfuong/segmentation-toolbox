import collections.abc
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.utils import to_ntuple, to_2tuple
# from mmcv.cnn.bricks import build_conv_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
# from mmcv.cnn.bricks.transformer import PatchEmbed

from timm.models.layers import DropPath, to_2tuple, trunc_normal_, create_conv2d, create_pool2d, _assert
from timm.models.helpers import named_apply
from ..builder import BACKBONES
# from mmcv.utils.conv import create_conv2d, create_conv2d_pad

#lass ConvPool(nn.Module):
#   def __init__(self, in_channels, out_channels, norm_layer, pad_type=''):
#       super().__init__()
#       self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=pad_type, bias=True)
#       self.norm = norm_layer(out_channels)
#       self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=pad_type)
#   
#   def forward(self,x):
#       assert(x.shape[-2]%2 == 0), 'BlockAgg requires even input spatial dims'
#       assert(x.shape[-1]%2 == 0), 'BlockAgg requires even input spatial dims'
#       x = self.conv(x)
#       x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
#       x = self.pool(x)
#       return x    # (B,C,H//2,W//2)

class ConvPool(BaseModule):
    def __init__(self, in_channels, out_channels, norm_layer, pad_type=''):
        super().__init__()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True)
        self.norm = norm_layer(out_channels)
        self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        _assert(x.shape[-2] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        _assert(x.shape[-1] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)


def blockify(x, block_size:int):
    B, H, W, C  = x.shape
    assert(H % block_size == 0), '`block_size` must divide input height evenly'
    assert(W % block_size == 0), '`block_size` must divide input width evenly'
    grid_height = H // block_size
    grid_width = W // block_size
    x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    return x  # (B, T, N, C)


def deblockify(x, block_size: int):
    B, T, _, C = x.shape
    grid_size = int(math.sqrt(T))
    height = width = grid_size * block_size
    x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x = x.transpose(2, 3).reshape(B, height, width, C)
    return x  # (B, H, W, C)


class Attention(BaseModule):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B (batch_size), T (image blocks), N (seq length per image block), C (embed dim)
        """ 
        B, T, N, C = x.shape
        # result of next line is (qkv, B, num (H)eads, T, N, (C')hannels per head)
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, H, T, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, T, N, C'), permute -> (B, T, N, C', H)
        x = (attn @ v).permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (B, T, N, C)


class Mlp(BaseModule):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(BaseModule):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert(H == self.img_size[0]), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert(W == self.img_size[1]), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TransformerLayer(BaseModule):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NestLevel(BaseModule):
    """ Single hierarchical level of a Nested Transformer
    """
    def __init__(
            self, 
            num_blocks, 
            block_size, 
            seq_length, 
            num_heads, 
            depth, 
            embed_dim, 
            prev_embed_dim=None,
            mlp_ratio=4., 
            qkv_bias=True, 
            drop_rate=0.,
            attn_drop_rate=0., 
            drop_path_rates=[],
            norm_layer=None, 
            act_layer=None, 
            pad_type=''):
        super().__init__()
        self.block_size = block_size
        self.grad_checkpointing = False

        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))

        if prev_embed_dim is not None:
            self.pool = ConvPool(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type)
        else:
            self.pool = nn.Identity()

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.transformer_encoder = nn.Sequential(*[
            TransformerLayer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rates[i],
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

    def forward(self, x):
        """
        expects x as (B, C, H, W)
        """
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
        x = blockify(x, self.block_size)  # (B, T, N, C')
        x = x + self.pos_embed
        # if self.grad_checpointing and not torch.jit.is_scripting():
            # x = checkpoint_seq(self.transformer_encoder, x)
        # else:
            # x = self.transformer_encoder(x)  # (B, T, N, C')
        x = self.transformer_encoder(x)  # (B, T, N, C')
        x = deblockify(x, self.block_size)  # (B, H', W', C')
        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 3, 1, 2)  # (B, C, H', W')

    
@BACKBONES.register_module()
class Nest(nn.Module):
    """
    Args:
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters
    """
    def __init__(
            self, 
            img_size=224, 
            patch_size=4, 
            in_chans=3,
            embed_dims=(128, 256, 512),
            depths=(2, 2, 20), 
            num_heads=(4, 8, 16), 
            num_levels=3, 
            num_classes=1000, 
            mlp_ratio=4., 
            qkv_bias=True,
            drop_rate=0., 
            attn_drop_rate=0., 
            drop_path_rate=0.5, 
            norm_layer=None, 
            act_layer=None,
            pad_type='same', 
            weight_init='', 
            global_pool='avg',
            out_indices=(0,1,2,3),
            frozen_stages=-1,
    ):
        for param_name in ['embed_dims', 'num_heads', 'depths']:
            param_value = locals()[param_name]

        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)

        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1]
            img_size = img_size[0]
        assert img_size % patch_size == 0, '`patch_size` must divide `img_size` evenly'
        self.patch_size = patch_size
        self.num_features = embed_dims[-1]
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.feature_info = []
        self.drop_rate = drop_rate
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.out_incides = out_indices

        # Num blocks per level
        self.num_blocks = (4 ** torch.arange(num_levels)).flip(0).tolist()
        assert(img_size // patch_size) % math.sqrt(self.num_blocks[0]) == 0, \
            'First level blocks don\'t fit evenly. Check `img_size`, `patch_size`, and `num_levels`.'

        # Block edge size in units of patches
        # Hint: (img_size // patch_size) gives number of patches along edge of image. sqrt(self.num_blocks[0]) is the
        #  number of blocks along edge of image
        self.block_size = int((img_size // patch_size) // math.sqrt(self.num_blocks[0]))
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], flatten=False)
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]
        # Added pos_drop after patch embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Build up each hierarchical level
        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        curr_stride = 4
        for i in range(len(self.num_blocks)):
            dim = embed_dims[i]
            levels.append(NestLevel(
                self.num_blocks[i], self.block_size, self.seq_length, num_heads[i], depths[i], dim, prev_dim,
                mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dp_rates[i], norm_layer, act_layer, pad_type=pad_type))
            self.feature_info += [dict(num_chs=dim, reduction=curr_stride, module=f'levels.{i}')]
            prev_dim = dim
            curr_stride *= 2
        self.levels = nn.Sequential(*levels)

        # Final normalization layer
        self.norm = norm_layer(embed_dims[-1])

        # Classifier
        # self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(weight_init)

    def forward_features(self, x):
        x = self.patch_embed(x)
        # Added forward pos_drop
        x = self.pos_drop(x)
        x = self.levels(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x
        # return tuple(x) ???
    
    def forward_head(self, x, pre_logits: bool=False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x
    
    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        for level in self.levels:
            trunc_normal_(level.pos_embed, std=.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    # Added _freeze_stages according to swin transformer
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.levels[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        

    def train(self, mode=True):
        super(Nest, self).train(mode)
        self._freeze_stages()

def _init_nest_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)




    
