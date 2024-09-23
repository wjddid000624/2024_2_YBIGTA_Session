import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class EmbeddingLayer(nn.Module):
    def __init__(self, num_patches, emb_dim, input_dim = 65):
        super(EmbeddingLayer, self).__init__()
        self.num_patches = num_patches
        self.emb_dim = emb_dim
        self.input_dim = input_dim

        self.projection = nn.Conv2d(input_dim, emb_dim, kernel_size=(5,1), stride=5)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # make sure that position embedding has same size with x
        position_embedding = nn.Parameter(torch.rand(b, self.num_patches + 1, self.emb_dim))

        x += position_embedding

        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads) ** -0.5
        
        self.q_layer = nn.Linear(emb_dim, emb_dim)
        self.k_layer = nn.Linear(emb_dim, emb_dim)
        self.v_layer = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x):
        q = rearrange(self.q_layer(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.k_layer(x), "b n (h d) -> b h n d", h=self.num_heads)
        v  = rearrange(self.v_layer(x), "b n (h d) -> b h n d", h=self.num_heads)

        q_shape = q.size(0)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        out = context.transpose(1, 2).contiguous().view(q_shape, -1, self.emb_dim)
        out = self.fc_out(out)

        return out
       
    
class MLP(nn.Module):
    def __init__(self, emb_dim, mlp_dim, dropout_ratio=0.1):
        super(MLP, self).__init__()
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim

        self.fc1 = nn.Linear(emb_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, emb_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)

        return x
    
class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, n_layers):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers

        layers = []
        for _ in range(self.n_layers):
            layers.append(nn.LayerNorm(emb_dim))
            layers.append(MultiHeadSelfAttention(emb_dim, num_heads))
            layers.append(nn.LayerNorm(emb_dim))
            layers.append(MLP(emb_dim, mlp_dim))

        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

#여기까지 Encoder 구현 끝!!


class VisionTransformer(nn.Module):
    def __init__(self, num_classes = 3, num_patches = 13, emb_dim = 768, 
                 num_heads = 12, mlp_dim = 3072, n_layers = 6):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers

        self.embedding = EmbeddingLayer(num_patches, emb_dim)
        self.blocks = Block(emb_dim, num_heads, mlp_dim, n_layers)
        self.norm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.fc(x)
        
        return x
    
