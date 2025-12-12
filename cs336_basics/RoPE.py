import torch 
from torch import nn
from einops import einsum
import math

class RoPE(nn.Module): 
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        positions = torch.arange(max_seq_len, device=device)
        inv_freq = theta ** -(torch.arange(0, d_k, 2, device=device) / d_k)
        angles = einsum(positions, inv_freq, "L, K -> L K")
        self.register_buffer("cos", torch.cos(angles), persistent=False, device=device)
        self.register_buffer("sin", torch.sin(angles), persistent=False, device=device)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        cos = self.cos[token_positions] # cos.shape == [... max_seq_len K]
        sin = self.sin[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_even_out = x_even * cos - x_even * sin
        x_odd_out = x_odd * cos + x_odd * sin
        x_empty = torch.empty_like(x)
        x_empty[..., 0::2] = x_even_out
        x_empty[..., 1::2] = x_odd_out
        return x_empty



