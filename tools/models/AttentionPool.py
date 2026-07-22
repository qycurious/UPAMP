import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        B, N, C = x.shape

        q = self.query(x)  # (B, N, C)
        k = self.key(x)  # (B, N, C)
        v = self.value(x)  # (B, N, C)

        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)  # (B, N, N)
        attn_output = attn_weights @ v  # (B, N, C)

        pooled = attn_output.mean(dim=1, keepdim=True)  # (B, 1, C)
        pooled = self.proj(pooled)  # (B, 1, C)

        return pooled