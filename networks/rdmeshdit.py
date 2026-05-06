### R-DMesh DiT 

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

def init_linear_xavier(m: nn.Linear, gain=1.0):
    nn.init.xavier_uniform_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    
class MLP(nn.Module):
    def __init__(self, device, dtype, width, init_scale=0.25):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        heads,
        init_scale=0.25,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention_flash2(heads=heads)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x
    
class QKVMultiheadAttention_flash2(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(bs, n_ctx, -1)
        return output

class QKVMultiheadCrossAttention_flash2(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, q, context):
        bs, n_ctx, width = q.shape
        n_ctx_context = context.shape[1]
        attn_ch = width // self.heads 
        q = q.view(bs, n_ctx, self.heads, -1)
        context = context.view(bs, n_ctx_context, self.heads, -1)
        k, v = torch.split(context, attn_ch, dim=-1)
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(bs, n_ctx, -1)
        return output
    
class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        heads,
        init_scale=0.25,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_q = nn.Linear(width, width, device=device, dtype=dtype)
        self.c_kv = nn.Linear(width, width * 2, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadCrossAttention_flash2(heads=heads)
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, context):
        x = self.c_q(x)
        context = self.c_kv(context)
        x = self.attention(x, context=context)
        x = self.c_proj(x)
        return x

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim, device="cuda", dtype=torch.float32, bias=True):
        super().__init__()
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias, device=device)
        self.norm = nn.LayerNorm(embedding_dim, device=device, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
class RDMeshAttentionBlock(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        heads,
        init_scale=1.0,
    ):
        super().__init__()
        
        
        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.adaln_x = AdaLayerNormZero(width, device=device, dtype=dtype)
        self.ln_x = nn.LayerNorm(width, device=device, dtype=dtype)

        self.ln_x_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.attn_vid = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
    
    def forward(self, x, t_emb, vid_emb):
        return self._forward_fixt(x, t_emb, vid_emb)

    def _forward_fixt(self, x, t_emb, vid_emb):
        
        # Vid CA
        x = self.ln_x_pre(x)
        vid_attn_out = self.attn_vid(x, vid_emb) # vid_emb has been normalized
        x = x + vid_attn_out

        # DMesh SA
        norm_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaln_x(x, emb=t_emb)
        norm_all = norm_x
        attn_output = self.attn(norm_all)
        attn_output_x = attn_output
        attn_output_x = gate_msa_x.unsqueeze(1) * attn_output_x
        x = x + attn_output_x
        
        # FFN
        norm_x = self.ln_x(x)
        norm_x = norm_x * (1 + scale_mlp_x[:, None]) + shift_mlp_x[:, None]
        ff_output = self.mlp(norm_x)
        ff_output_x = ff_output
        ff_output_x = gate_mlp_x[:, None] * ff_output_x
        x = x + ff_output_x

        return x
    
class Transformer_rdmesh(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        layers,
        heads,
        init_scale=0.25,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                RDMeshAttentionBlock(
                    device=device,
                    dtype=dtype,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, t_emb, vid_emb):
        for block in self.resblocks:
            x = block(x, t_emb, vid_emb=vid_emb)
        return x

class RDMeshDiT(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        input_channels=3*16,
        output_channels=3*16,
        width=512,
        layers=12,
        heads=8,
        init_scale=0.25,
        dit_layers=1,
        **kwargs,
    ):
        super().__init__()
        
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.backbone = Transformer_rdmesh(
            device=device,
            dtype=dtype,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale
        )

        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_pre_vid = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        self.dit_proj = nn.Linear(3072*dit_layers, width, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def _forward_vc(self, x, t_emb, vid_emb):
        h = self.input_proj(x)  
        h = self.ln_pre(h)
        vid_emb = self.ln_pre_vid(vid_emb)
        h = self.backbone(h, t_emb=t_emb, vid_emb=vid_emb)
        h = self.ln_post(h)
        h = self.output_proj(h)
        return h
    
    def forward(self, x, t, vid_embed):

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        vid_embed = self.dit_proj(vid_embed)

        return self._forward_vc(x, t_embed, vid_emb=vid_embed)
