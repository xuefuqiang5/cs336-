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
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.max_seq_len = max_seq_len
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor: 
        L = x.size(-2)
        device = x.device
        if token_positions == None: 
            token_positions = torch.arange(0, L, device=device)

        cos = self.cos[token_positions] # cos.shape == [... max_seq_len K]
        sin = self.sin[token_positions]
        cos = cos.to(device)
        sin = sin.to(device)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_out = x_even * cos - x_odd * sin
        x_odd_out  = x_even * sin + x_odd * cos
        x_empty = torch.empty_like(x)
        x_empty[..., 0::2] = x_even_out
        x_empty[..., 1::2] = x_odd_out
        return x_empty



