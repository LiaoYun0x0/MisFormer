import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torchvision as tv
from functools import partial
from .attention import LinearAttention,FullAttention,TopKWindowAttention

AFFINE = True
TRACK_RUNNING_STATs = True
NORM = nn.BatchNorm2d


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class ConvBNGelu(nn.Module):
    def __init__(self,c_in,c_out,k,s):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,c_out,k,s,k//2,bias=False),
            NORM(c_out,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs),
            nn.GELU()
        )
    def forward(self,x):
        return self.net(x)
    
class MB(nn.Module):
    def __init__(self,dim_in,dim_out,mlp_ratio=4,prenorm=False,afternorm=True,k=3,stride=1):
        super().__init__()
        dim_mid = int(dim_in * mlp_ratio)
        self.net = nn.Sequential(
            NORM(dim_in,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs) if prenorm else nn.Identity(),
            nn.Conv2d(dim_in, dim_mid, 1,bias=False),
            NORM(dim_mid,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_mid, k,stride,k//2,groups=dim_mid,bias=False),
            NORM(dim_mid,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_out, 1,bias=False),
            NORM(dim_out,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs) if afternorm else nn.Identity()
        )
    def forward(self,x):
        x = self.net(x)
        return x


class ResidualMB(nn.Module):
    def __init__(self,dim_in,dim_out,mlp_ratio=4,prenorm=False,afternorm=True,k=3,stride=1,dropout=0.):
        super().__init__()
        dim_mid = int(dim_in * mlp_ratio)
        self.net = nn.Sequential(
            NORM(dim_in,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs) if prenorm else nn.Identity(),
            nn.Conv2d(dim_in, dim_mid, 1,bias=False),
            NORM(dim_mid,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_mid, k,stride,k//2,groups=dim_mid,bias=False),
            NORM(dim_mid,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs),
            nn.GELU(),
            nn.Conv2d(dim_mid, dim_out, 1,bias=False),
            NORM(dim_out,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs) if afternorm else nn.Identity()
        )
        self.main = nn.Sequential(
            nn.MaxPool2d(k, stride, k//2) if stride > 1 else nn.Identity(),
            nn.Conv2d(dim_in, dim_out, 1, bias=False) if dim_in != dim_out else nn.Identity()
        )
        self.dropout = DropPath(dropout)
            
    def forward(self,x):
        return self.main(x) + self.dropout(self.net(x))
        
    
class ConvBlock(nn.Module):
    def __init__(self,dim,dropout=0.,mlp_ratio=4):
        super().__init__()
        self.conv = MB(dim,dim,mlp_ratio,False,True)
        self.mlp = MB(dim,dim,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
    def forward(self,x):
        x = x + self.dropout(self.conv(x))
        x = x + self.dropout(self.mlp(x))
        return x
       
class TopKWindowAttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,w=7,k=8,dropout=0.0,attention='linear',mlp_ratio=4):
        super(TopKWindowAttentionLayer, self).__init__()
        self.w = w
        self.k = k
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        
        self.qkv_proj = nn.Sequential(
            NORM(d_model,affine=AFFINE,track_running_stats=TRACK_RUNNING_STATs),
            nn.GELU(),
            nn.Conv2d(d_model, 3*d_model, 1,1,bias=False)    
        )
        if w == 1:
            if attention == 'linear':
                self.attention = LinearAttention()
            elif attention == 'full':
                self.attention = FullAttention()
            else:
                raise NotImplementedError()
        else:
            self.attention = TopKWindowAttention(d_head,w=w,k=k,attention=attention)
        self.merge = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
        )
        self.mlp = MB(d_model,d_model,mlp_ratio,False,True)
        self.dropout = DropPath(dropout)
        
    def forward(self, x):
        b,d,h,w = x.shape
        q,k,v = torch.chunk(self.qkv_proj(x), 3,dim=1)
        
        if self.w == 1:
            q = rearrange(q, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
            k = rearrange(k, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
            v = rearrange(v, ' b (heads d) h w -> b (h w) heads d', heads=self.nhead)
            message = self.attention(q, k, v, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
            message = rearrange(message,'b (h w) heads d -> b (heads d) h w',h=h)
        else:
            message = self.attention(q,k,v)
        
        x = x + self.dropout(self.merge(message))
        x = x + self.dropout(self.mlp(x))
        return x

class AttentionLayer(nn.Module):
    def __init__(self,d_model,d_head,dropout=0.0,attention='linear',mlp_ratio=4):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.nhead = self.d_model // self.d_head
        
        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(d_model,elementwise_affine=AFFINE),
            nn.GELU(),
            nn.Linear(d_model,3*d_model,bias=False)  
        )
        if attention == 'linear':
            self.attention = LinearAttention()
        elif attention == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError()
        
        self.merge = nn.Sequential(
            nn.Linear(d_model, d_model)
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*mlp_ratio, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = DropPath(dropout)
        
    def forward(self, x):
        q,k,v = torch.chunk(self.qkv_proj(x), 3,dim=-1)
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.nhead)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.nhead)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.nhead)
        message = self.attention(q, k, v, q_mask=None, kv_mask=None)  # [N, L, (H, D)]
        message = rearrange(message,'b l h d -> b l (h d)')
        
        x = x + self.dropout(self.merge(message))
        x = x + self.dropout(self.mlp(x))
        return x

class TopKWindowViT_meta(nn.Module):
    def __init__(
        self,
        *,
        dims=[128,192,256,320],
        depths=[2,2,6,2],
        depths2=[(2,0),(1,1),(1,3),(0,2)],
        dim_head = 32,
        dim_conv_stem = None,
        window_size = [8,8,4,2],
        ks = [8,6,4,2],
        num_attn = 5,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        in_chans = 22,
        num_classes = 2,
    ):
        super().__init__()
        assert isinstance(depths, list), 'depth needs to be list if integers indicating number of transformer blocks at that stage'

        # convolutional stem
        if dim_conv_stem is None:
            dim_conv_stem = 64

        self.conv_stem = nn.Sequential(
            ConvBNGelu(in_chans, dim_conv_stem, 3, 2),
            ConvBNGelu(dim_conv_stem, dim_conv_stem, 3, 1),
            ConvBNGelu(dim_conv_stem, dim_conv_stem, 3, 1),
            ConvBNGelu(dim_conv_stem, dim_conv_stem, 3, 2)
        )

        num_stages = len(depths)
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.layers = nn.ModuleList([])
        
        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            self.layers.append(nn.ModuleList())
            stage = self.layers[ind]
            s = (2 if ind!=0 else 1)
            stage.append(ResidualMB(layer_dim_in,layer_dim,mbconv_expansion_rate,True,True,stride=s,dropout=dropout))
            k = ks[ind]
            w = window_size[ind]
            for stage_ind in range(layer_depth):
                if ind == 0:
                    block = ConvBlock(layer_dim,dropout=dropout,mlp_ratio=mbconv_expansion_rate)
                else:
                    if stage_ind == 0:
                        block = ConvBlock(layer_dim,dropout=dropout,mlp_ratio=mbconv_expansion_rate)
                    else:
                        block = TopKWindowAttentionLayer(layer_dim, dim_head,w=w,k=k,dropout=dropout)
                        # block = ConvBlock(layer_dim,dropout=dropout,mlp_ratio=mbconv_expansion_rate)
                stage.append(block)
        
        # for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
        #     self.layers.append(nn.ModuleList())
        #     stage = self.layers[ind]
        #     s = (2 if ind!=0 else 1)
        #     stage.append(ResidualMB(layer_dim_in,layer_dim,mbconv_expansion_rate,True,True,stride=s,dropout=dropout))
        #     k = ks[ind]
        #     w = window_size[ind]
        #     d1,d2 = depths2[ind]
        #     for i,stage_ind in enumerate(range(layer_depth)):
        #         block = nn.Sequential()
        #         for _ in range(d1):
        #             block.append(ConvBlock(layer_dim,dropout=dropout,mlp_ratio=mbconv_expansion_rate))
        #         for _ in range(d2):
        #             block.append(TopKWindowAttentionLayer(layer_dim, dim_head,w=w,k=k,dropout=dropout))
        #         stage.append(block)
        
        self.seed_shape_linear = []
        depths = [15] + list(dims[1:])
        for i in range(1,len(depths)):
            self.seed_shape_linear.extend([
                nn.Linear(depths[i-1], depths[i]),
                nn.LayerNorm(depths[i],elementwise_affine=AFFINE) if i < len(depths)-1 else nn.Identity(),
                nn.GELU() if i < len(depths)-1 else nn.Identity()
            ])
        self.seed_shape_linear = nn.Sequential(*self.seed_shape_linear)
        
        self.seed_light_linear = []
        depths = [19] + list(dims[1:])
        # depths = [10] + [256,320,384,448,512]
        for i in range(1,len(depths)):
            self.seed_light_linear.extend([
                nn.Linear(depths[i-1], depths[i]),
                nn.LayerNorm(depths[i],elementwise_affine=AFFINE) if i < len(depths)-1 else nn.Identity(),
                nn.GELU() if i < len(depths)-1 else nn.Identity()
            ])
        self.seed_light_linear = nn.Sequential(*self.seed_light_linear)
        
        self.attn = nn.Sequential()
        for i in range(num_attn):
            self.attn.append(AttentionLayer(dims[-1],dim_head,attention='linear'))
            
        self.pos_embedding = nn.Parameter(torch.randn(1, 33, dims[-1]))
        self.cls_token = nn.Parameter(torch.randn(1,1,dims[-1]))
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dims[-1],elementwise_affine=AFFINE),
            nn.Linear(dims[-1], dims[-1]*mbconv_expansion_rate),
            nn.GELU(),
            nn.Linear(dims[-1]*mbconv_expansion_rate, num_classes)
        )
        
    def forward(self, data):
        # x = rearrange(data['seed_image'],'b ns c h w -> b (ns c) h w')
        x = data['seed_image'].permute(0,3,1,2)
        x = self.conv_stem(x)
        for i,stage in enumerate(self.layers):
            for _stage in stage:
                x = _stage(x)
        x = self.pool(x).view(x.shape[0],x.shape[1])
        
        # x_shape = self.seed_shape_linear(data['seed_shape'])
        # x = x + x_shape
        
        # x_light = data['seed_light'][...,0]
        # x_light = self.seed_light_linear(x_light)
        # x = x + x_light
        
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.backbone = TopKWindowViT_meta(**config)

    def forward(self,x):
        return self.backbone(x)
    
    def forward_train(self,data):
        logit = self.backbone(data)
        loss = F.cross_entropy(logit, data['seed_label'])
        return loss
    