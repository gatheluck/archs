from typing import Final, List

import torch
import torch.nn as nn
from einops import rearrange


class MultiheadSelfAttention(nn.Module):
    """Multihead self-attention (MSA) module.

    This class is implementation of Multihead self-attention (MSA) module.
    If the num_heads = 1, the result is same as standard self attention.
    In each head, Scaled Dot-Product Attention is used.

    Note:
        Nortation for docstring and comments:
            B: The size of batch.
            N: The length of sequence.
            D: The dimension of latent vector.
            Dh: The dimension of each head (Usually, = D / num_head).

    Attributes:
        num_head (int): The number of head. If this value is 1, the result is same as standard self attention.
        scale (floar): The saling factor of attention weight.
        to_qkv (nn.Module): The map from input to multihead qkv.
        to_out (nn.Module): The map concatenated output from each head to the original dimension.

    """

    def __init__(
        self,
        dim: int,
        num_head: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        add_bias_qkv: bool = False,
    ) -> None:
        """

        Args:
            dim (int): The dimension of original latent (=D).
            num_head (int): The number of head. If this value is 1, the result is same as standard self attention.
            dim_head (int): The dimension of each head (=Dh).
            dropout (float): The ratio of dropout which is applied just before return.
            add_bias_qkv (bool): If True, add bias term to to_qkv.

        """
        super().__init__()
        self.num_head: Final[int] = num_head
        self.scale: Final[float] = dim_head ** -0.5  # =sqrt(Dh)

        dim_inner: Final[int] = num_head * dim_head
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=add_bias_qkv)
        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim), nn.Dropout(dropout))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """

        Args:
            z (torch.Tensor): An input size of (B, N, D).

        Rreturns:
            torch.Tensor: An output size of (B, N, D).

        """
        # Calculte q, k, v in each head.
        qkv: List[torch.Tensor] = self.to_qkv(z).chunk(
            3, dim=-1
        )  # list of (B, N, num_head * dim_head)
        q, k, v = [
            rearrange(t, "b n (h d) -> b h n d", h=self.num_head) for t in qkv
        ]  # list of (B, num_head, N, dim_head)

        # Calulate attention weight in each head (Scaled Dot-Product Attention).
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # (B, num_head, N, N)
        attn = dots.softmax(dim=-1)  # (B, num_head, N, N)

        # Calulate self attention in each head.
        self_atten = torch.einsum(
            "bhij,bhjd->bhid", attn, v
        )  # (B, num_head, N, dim_head)
        self_atten = rearrange(
            self_atten, "b h n d -> b n (h d)"
        )  # (B, N, num_head * dim_head)

        # Project concatenated output from each head.
        return self.to_out(self_atten)  # (B, N, D)
