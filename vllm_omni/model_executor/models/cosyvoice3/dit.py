"""
in notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from diffusers.models.normalization import AdaLayerNormZero
from einops import repeat
from torch import nn
from vllm.logger import init_logger
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

# rotary positional embedding related

logger = init_logger(__name__)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int | None = None,  # if not None -> joint attention
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if self.context_dim is not None:
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = nn.Linear(context_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, dim)

    def forward(
        self,
        x: torch.Tensor,  # noised input x  # noqa: F722
        c: torch.Tensor | None = None,  # context c  # noqa: F722
        mask: torch.Tensor | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor


class AttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: torch.Tensor,  # noised input x  # noqa: F722
        mask: torch.Tensor | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            else:
                mask = mask[:, 0, -1].unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# DiT Block


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):  # noqa: F722
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv1(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv2(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


# Text embedding
def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    # NOTE this modified implementation meets onnx export requirements, but it doesn't support num_left_chunks
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode="trunc") + 1) * chunk_size
    ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    return ret


def add_optional_chunk_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
):
    """Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        chunk_masks (torch.Tensor): chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1,)).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size, num_left_chunks, xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size, num_left_chunks, xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    assert chunk_masks.dtype == torch.bool
    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        logger.debug(
            "get chunk_masks all false at some timestep, force set to true, "
            "make sure they are masked in futuer computation!"
        )
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    return chunk_masks


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: torch.Tensor, seq_len, drop_text=False):  # noqa: F722
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, spks):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = repeat(spks, "b c -> b t c", t=x.shape[1])
            to_cat.append(spks)

        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        if streaming is True:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, self.static_chunk_size, -1).unsqueeze(
                dim=1
            )
        else:
            attn_mask = (
                add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1).unsqueeze(dim=1)
            )

        for block in self.transformer_blocks:
            x = block(x, t, mask=attn_mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output
