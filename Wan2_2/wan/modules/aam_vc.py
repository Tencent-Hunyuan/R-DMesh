### Video-cond traj generation model

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.no_grad()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@torch.no_grad()
def rope_appl_t(x, grid_sizes, freqs):
    assert x.dim() == 4, "x must be [B, S, N, D]"
    B, S, N, D = x.shape
    assert D % 2 == 0, "D must be even for complex pairing"
    c = D // 2
    if isinstance(grid_sizes, torch.Tensor):
        gs = grid_sizes.detach().cpu().tolist()
    else:
        gs = grid_sizes
    # freqs_t: [max_T, c] complex, move to x.device
    freqs_t = freqs.to(x.device)
    x_dtype = x.dtype
    x_out = x.clone()
    for i, (T, H, W) in enumerate(gs):
        seq_len = T * H * W
        if seq_len == 0:
            continue
        xi = x_out[i, :seq_len].to(torch.float32)
        xi_complex = torch.view_as_complex(xi.reshape(seq_len, N, c, 2))
        t_idx = torch.arange(seq_len, device=x.device) // (H * W)  # [seq_len]
        freqs_pos = freqs_t[t_idx, :]  # uses only T-axis RoPE
        freqs_pos = freqs_pos.unsqueeze(1)
        xi_rot = xi_complex * freqs_pos  # [seq_len, N, c]
        xi_real = torch.view_as_real(xi_rot).reshape(seq_len, N, D).to(x_dtype)
        x_out[i, :seq_len] = xi_real
    return x_out

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

class QKVMultiheadAttention(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

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

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        heads,
        init_scale=0.25,
        use_flash2=True,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        if use_flash2:
            self.attention = QKVMultiheadAttention_flash2(heads=heads)
        else:
            self.attention = QKVMultiheadAttention(heads=heads)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x

class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, q, context):
        bs, n_ctx, width = q.shape
        n_ctx_context = context.shape[1]
        attn_ch = width // self.heads 
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        context = context.view(bs, n_ctx_context, self.heads, -1)
        k, v = torch.split(context, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

class QKVMultiheadCrossAttention_flash2(nn.Module):
    def __init__(self, width, heads, device):
        super().__init__()
        self.heads = heads
        d = width // heads 
        c = d // 2
        self.freqs = torch.cat([
            rope_params(1024, 2*(c - 2 * (c // 3))),
            rope_params(1024, 2*(c // 3)),
            rope_params(1024, 2*(c // 3))
        ], dim=1).to(device)

    def forward(self, q, context, apply_rope=False, grid_sizes=None):
        bs, n_ctx, width = q.shape
        n_ctx_context = context.shape[1]
        attn_ch = width // self.heads 
        q = q.view(bs, n_ctx, self.heads, -1)
        context = context.view(bs, n_ctx_context, self.heads, -1)
        k, v = torch.split(context, attn_ch, dim=-1)
        if apply_rope:
            k = rope_apply(k, grid_sizes, self.freqs)
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
        context_width=None,
        init_scale=0.25,
        use_flash2=True,
    ):
        super().__init__()
        if context_width is None:
            context_width = width
        self.width = width
        self.heads = heads
        self.c_q = nn.Linear(width, width, device=device, dtype=dtype)
        self.c_kv = nn.Linear(context_width, width * 2, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        # if use_flash2:
        self.attention = QKVMultiheadCrossAttention_flash2(width=width, heads=heads, device=device)
        # else:
        #     self.attention = QKVMultiheadCrossAttention(heads=heads)
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, context, apply_rope=False, grid_sizes=None):
        x = self.c_q(x)
        context = self.c_kv(context)
        x = self.attention(x, context=context, apply_rope=apply_rope, grid_sizes=grid_sizes)
        x = self.c_proj(x)
        return x

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim, device="cuda", dtype=torch.float32, norm_type="layer_norm", bias=True):
        super().__init__()
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias, device=device)

        self.norm = nn.LayerNorm(embedding_dim, device=device, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
class AdaLN_Fusion(nn.Module):
    """
    An adaptive Layer Normalization module that generates all necessary 
    scale, shift, and gate parameters for a full Transformer block 
    (Cross-Attention, Self-Attention, and MLP/FFN).

    It generates 9 parameters in total.
    """
    def __init__(self, embedding_dim, bias=True, device="cuda", dtype=torch.float32):
        super().__init__()
        
        self.silu = nn.SiLU()
        # The linear layer now outputs 9 sets of parameters.
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias, device=device, dtype=dtype)
        
        # This norm will be used for pre-normalization before each sub-layer.
        # self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)

    def forward(self, emb):
        """
        Takes the timestep embedding and generates all modulation parameters.
        Note: This module no longer processes the main data tensor 'x'.
              It only generates parameters based on 'emb'.

        Args:
            emb (torch.Tensor): The timestep embedding tensor of shape [B, C].

        Returns:
            dict: A dictionary containing 9 parameter tensors, each of shape [B, C].
        """
        # Pass the embedding through SiLU and the linear layer
        params = self.linear(self.silu(emb))
        
        # Split the output into 9 individual parameter tensors
        (
            shift_cross, scale_cross, gate_cross,
            shift_self,  scale_self,  gate_self,
            shift_mlp,   scale_mlp,   gate_mlp
        ) = params.chunk(9, dim=1)
        
        # Return them in a structured dictionary for clarity
        return {
            "cross": {"shift": shift_cross, "scale": scale_cross, "gate": gate_cross},
            "self":  {"shift": shift_self,  "scale": scale_self,  "gate": gate_self},
            "mlp":   {"shift": shift_mlp,   "scale": scale_mlp,   "gate": gate_mlp},
        }

class AdaLN_Wan(nn.Module):
    """
    An adaptive Layer Normalization module that generates all necessary 
    scale, shift, and gate parameters for a full Transformer block 
    (Cross-Attention, Self-Attention, and MLP/FFN).

    It generates 9 parameters in total.
    """
    def __init__(self, embedding_dim, output_dim=None, bias=True, device="cuda", dtype=torch.float32):
        super().__init__()
        
        if output_dim is None:
            output_dim = embedding_dim

        self.silu = nn.SiLU()
        # The linear layer now outputs 9 sets of parameters.
        self.linear = nn.Linear(embedding_dim, 3 * output_dim, bias=bias, device=device, dtype=dtype)
        
        # This norm will be used for pre-normalization before each sub-layer.
        # self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6, device=device, dtype=dtype)

    def forward(self, emb):
        """
        Takes the timestep embedding and generates all modulation parameters.
        Note: This module no longer processes the main data tensor 'x'.
              It only generates parameters based on 'emb'.

        Args:
            emb (torch.Tensor): The timestep embedding tensor of shape [B, C].

        Returns:
            dict: A dictionary containing 9 parameter tensors, each of shape [B, C].
        """
        # Pass the embedding through SiLU and the linear layer
        params = self.linear(self.silu(emb))
        
        # Split the output into 9 individual parameter tensors
        (
            shift_cross, scale_cross, gate_cross
        ) = params.chunk(3, dim=1)
        
        # Return them in a structured dictionary for clarity
        return {
            "cross": {"shift": shift_cross, "scale": scale_cross, "gate": gate_cross},
        }

def initialize_weights(m):
    """
    Initializes weights of Linear and LayerNorm layers.
    """
    if isinstance(m, nn.Linear):
        # Apply Xavier Uniform initialization to linear layer weights
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            # Initialize bias to zero
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        # Initialize LayerNorm bias to zero and weight to one
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1.0)

class CogXAttentionBlock(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        heads,
        init_scale=1.0,
        use_flash2=True,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            use_flash2=use_flash2,
        )
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.param_generator = AdaLN_Fusion(width, device=device, dtype=dtype)
        self.param_generator_vid = AdaLN_Wan(3072, device=device, dtype=dtype)
        
        self.norm_cross = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_self  = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm_mlp   = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)

        self.norm_cross_vid = nn.LayerNorm(3072, elementwise_affine=False, eps=1e-6)

        self.cross_attn_vid = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            context_width=3072,
        )
        
        ###
        self.cross_attn_4d = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            width=3072,
            heads=heads,
            init_scale=init_scale,
            use_flash2=use_flash2,
            context_width=width,
        )
    
        with torch.no_grad():
            self.apply(initialize_weights)
            # AdaLN
            torch.nn.init.constant_(self.param_generator.linear.weight, 0)
            if self.param_generator.linear.bias is not None:
                torch.nn.init.constant_(self.param_generator.linear.bias, 0)
            torch.nn.init.constant_(self.param_generator_vid.linear.weight, 0)
            if self.param_generator.linear.bias is not None:
                torch.nn.init.constant_(self.param_generator_vid.linear.bias, 0)

    def forward(self, x, t_emb, vid_emb, t_emb_wan):
        
        ada_params = self.param_generator(t_emb)
        ada_params_vid = self.param_generator_vid(t_emb_wan)

        # Cross Attention
        x_norm = self.norm_cross(x)
        x_mod = x_norm * (1 + ada_params["cross"]["scale"][:, None]) + ada_params["cross"]["shift"][:, None]
        
        vid_norm = self.norm_cross_vid(vid_emb)
        vid_mod = vid_norm * (1 + ada_params_vid["cross"]["scale"][:, None]) + ada_params_vid["cross"]["shift"][:, None]
       
        attn_out_x = self.cross_attn_vid(x_mod, vid_mod)
        attn_out_vid = self.cross_attn_4d(vid_mod, x_mod)

        x = x + ada_params["cross"]["gate"][:, None] * attn_out_x
        vid_emb = vid_emb + ada_params_vid["cross"]["gate"][:, None] * attn_out_vid

        # Self Attention 
        x_norm = self.norm_self(x)
        x_mod = x_norm * (1 + ada_params["self"]["scale"][:, None]) + ada_params["self"]["shift"][:, None]
        attn_out = self.attn(x_mod)
        x = x + ada_params["self"]["gate"][:, None] * attn_out

        # FFN
        x_norm = self.norm_mlp(x)
        x_mod = x_norm * (1 + ada_params["mlp"]["scale"][:, None]) + ada_params["mlp"]["shift"][:, None]
        mlp_out = self.mlp(x_mod)
        x = x + ada_params["mlp"]["gate"][:, None] * mlp_out

        return x, vid_emb

class Transformer_cogx(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        width,
        layers,
        heads,
        init_scale=0.25,
        use_flash2=True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                CogXAttentionBlock(
                    device=device,
                    dtype=dtype,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    use_flash2=use_flash2,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, t_emb, vid_emb, t_embed_wan):
        for block in self.resblocks:
            x, vid_emb = block(x, t_emb, vid_emb=vid_emb, t_embed_wan=t_embed_wan)
        return x, vid_emb

class DyMeshVCDiT_joint(nn.Module):
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
        use_flash2=True,
        cond_drop_prob=0.0,
        **kwargs,
    ):
        super().__init__()

        self.cond_drop_prob = cond_drop_prob
        
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.time_embed_wan = MLP(
            device=device, dtype=dtype, width=3072, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.backbone = Transformer_cogx(
            device=device,
            dtype=dtype,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            use_flash2=use_flash2,
        )

        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)

        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def _forward_vc(self, x, t_emb, vid_emb, t_embed_wan):
        h = self.input_proj(x)  
        h = self.ln_pre(h)
        h, vid_emb = self.backbone(h, t_emb=t_emb, vid_emb=vid_emb, t_embed_wan=t_embed_wan)
        h = self.ln_post(h)
        h = self.output_proj(h)
        return h, vid_emb
    
    def prepare_t_emb(self, t):
        return self.time_embed(timestep_embedding(t, self.backbone.width)), self.time_embed_wan(timestep_embedding(t, 3072))

    def forward(self, x, t, wan_cond_latent):
        t_embed, t_embed_wan = self.prepare_t_emb(t)
        return self._forward_vc(x, t_embed, vid_emb=wan_cond_latent, t_embed_wan=t_embed_wan)
