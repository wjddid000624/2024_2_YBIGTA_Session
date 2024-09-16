import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        #TODO
        # Self attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()

        # Attention with encoder result
        self.src_attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = DropoutLayer(dropout)
        self.residual2 = ResidualConnection()

        self.ffn = FeedForwardLayer(d_model, d_ff)
        self.norm3 = LayerNormalization(d_model)
        self.dropout3 = DropoutLayer(dropout)
        self.residual3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        # Self attention
        att = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(self.residual1(x, self.dropout1(att)))

        # Attention with encoder result
        att = self.src_attn(x, memory, memory, src_mask)
        x = self.norm2(self.residual2(x, self.dropout2(att)))

        # Feed forward
        x = self.norm3(self.residual3(x, self.dropout3(self.ffn(x))))

        return x


        

            