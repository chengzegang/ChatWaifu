from typing import Callable
import torch
from torch.nn import Module, Parameter, TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
import torch.nn.functional as F


class VanillaTransformer(Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ff_size: int,
        dropout: float,
        activation: Callable = F.gelu,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                hidden_size, num_heads, ff_size, dropout, activation, batch_first=True
            ),
            num_layers,
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        return self.encoder(x, src_key_padding_mask=attn_mask)
