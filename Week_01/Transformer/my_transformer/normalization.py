import torch
import torch.nn as nn
from torch import Tensor

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(LayerNormalization, self).__init__()
        #TODO one line!
        self.d_model = d_model
    
    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        out = (x - mean) / (std + 1e-6)

        gamma = nn.Parameter(torch.ones(self.d_model))
        beta = nn.Parameter(torch.zeros(self.d_model))

        return gamma * out + beta