from torch import nn, Tensor
from typing import Optional
from .scaled_dot_product_attention import scaled_dot_product_attention


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int) -> None:
        super().__init__()

        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Optional[Tensor] = None,
        key_mask: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask
        )
        return attn_outputs
