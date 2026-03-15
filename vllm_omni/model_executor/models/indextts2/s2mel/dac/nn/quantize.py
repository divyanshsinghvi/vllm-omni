import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm_omni.model_executor.models.indextts2.s2mel.dac.nn.layers import WNConv1d


class VectorQuantize(nn.Module):
    """VQ implementation with factorized codes and l2-normalized lookup."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z, z_mask=None):
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        if z_mask is not None:
            commitment_loss = (F.mse_loss(z_e, z_q.detach(), reduction="none").mean(1) * z_mask).sum() / z_mask.sum()
            codebook_loss = (F.mse_loss(z_q, z_e.detach(), reduction="none").mean(1) * z_mask).sum() / z_mask.sum()
        else:
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z_e.detach())

        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices
