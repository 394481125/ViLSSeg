import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F

# from utils.mamba_uneter import UnetrUpBlock


# from utils.mamba_uneter import UnetrUpBlock

from monai.networks.blocks.unetr_block import UnetrUpBlock

from utils.vmamba import SS2D

from mamba_ssm import Mamba


# from utils.moe_layer.triton_src.moe_layer import MoE

# from utils.st_moe_pytorch.st_moe_pytorch import MoE as MoE_ST

class RMSNorm(torch.nn.Module):

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout=0, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)  # size = [batch, L, d_model]
        return self.dropout(x)  # size = [batch, L, d_model]


class BertAdapter(nn.Module):
    def __init__(self, hidden_size=768, adapter_size=64):
        super(BertAdapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        nn.init.normal_(self.down_project.weight, std=1e-2)
        nn.init.zeros_(self.down_project.bias)

        self.activation = nn.ReLU()

        self.up_project = nn.Linear(adapter_size, hidden_size)
        nn.init.normal_(self.up_project.weight, std=1e-2)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states: torch.Tensor):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        return hidden_states + up_projected


class GuideDecoderLayer(nn.Module):

    def __init__(self, in_channels: int, output_text_len: int, input_text_len: int = 24, embed_dim: int = 768):
        super(GuideDecoderLayer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        # self.self_attn_norm = RMSNorm(in_channels)

        self.self_attn_norm_text = nn.LayerNorm(in_channels)
        # self.self_attn_norm_text = RMSNorm(in_channels)

        self.cross_attn_norm = nn.LayerNorm(in_channels)
        # self.cross_attn_norm = RMSNorm(in_channels)

        self.self_attention_vision = SS2D(d_model=in_channels, dropout=0, d_state=16)

        # https://github.com/KyanChen/RSMamba

        self.self_attention_text = Mamba(d_model=in_channels, d_state=16)
        self.skip_scale_text = nn.Parameter(torch.ones(1))

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        # self.cross_attn_oe = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)

        self.adapter = BertAdapter(hidden_size=768, adapter_size=64)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm22 = nn.LayerNorm(in_channels)
        self.norm11 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        # self.norm1 = RMSNorm(in_channels)
        # self.norm_moe = RMSNorm(in_channels)
        # self.norm11 = RMSNorm(in_channels)
        # self.norm2 = RMSNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(1.421), requires_grad=True)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, txt, spatial_size):
        '''
        x:[B N C1]
        txt:[B,L,C]
        '''

        # txt = self.adapter(txt)

        txt = self.text_project(txt)

        txt2 = self.norm11(txt)

        txt2 = self.self_attention_text(txt2)

        txt2 = self.self_attn_norm_text(txt2)

        txt = self.skip_scale_text * txt + txt2

        txt = self.norm22(txt)

        # Self-Attention
        vis2 = self.norm1(x)
        #
        vis2 = rearrange(vis2, 'B (H W) C -> B H W C', H=spatial_size, W=spatial_size)

        # q = k = self.vis_pos(vis2)

        # vis2 = self.self_attn(q, k, value=vis2)[0]

        vis2_ori = self.self_attention_vision(vis2)

        vis2 = vis2_ori

        vis2 = rearrange(vis2, 'B H W C -> B (H W) C', H=spatial_size, W=spatial_size)

        vis2 = self.self_attn_norm(vis2)

        vis = x + vis2

        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2, _ = self.cross_attn(query=self.vis_pos(vis2),
                                  key=self.txt_pos(txt),
                                  value=txt)
        vis2 = self.cross_attn_norm(vis2)

        # vis2_moe,_ = self.sigma_moe_all(vis2)
        #
        # vis2_moe = self.norm_moe_all(vis2_moe)
        #
        # vis2 = vis2 + vis2_moe

        vis = vis + self.scale * vis2

        return vis


class GuideDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, spatial_size, text_len) -> None:
        super().__init__()

        self.guide_layer = GuideDecoderLayer(in_channels, text_len)  # for skip
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2, in_channels, out_channels, 3, 2, norm_name='BATCH')
        # self.decoder = UnetrUpBlock(in_channels,out_channels)

    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis = self.guide_layer(vis, txt, self.spatial_size)

        vis = rearrange(vis, 'B (H W) C -> B C H W', H=self.spatial_size, W=self.spatial_size)
        skip_vis = rearrange(skip_vis, 'B (H W) C -> B C H W', H=self.spatial_size * 2, W=self.spatial_size * 2)

        output = self.decoder(vis, skip_vis)

        output = rearrange(output, 'B C H W -> B (H W) C')

        return output


