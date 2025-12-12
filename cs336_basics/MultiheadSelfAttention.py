import torch 
from torch import nn
from einops import einsum, reduce
import math

def softmax(x: torch.Tensor, dimension: int) -> torch.Tensor: 
    x = x - x.max(dim=dimension, keepdim=True).values
    exp_x = torch.exp(x)
    t = torch.sum(exp_x, dim=dimension, keepdim=True)
    return exp_x/t


def scaled_dot_product_attention(
        q:torch.Tensor, 
        k:torch.Tensor, 
        v:torch.Tensor, 
        mask:torch.Tensor | None = None
    ) -> torch.Tensor:
    d_k = q.size(-1)
    attention_score = einsum(q, k, "b ... q_len d_k, b ... k_len d_k -> b ... q_len k_len") / math.sqrt(d_k)
    if mask != None: 
        attention_score = attention_score.masked_fill(mask==0, -1e9)
    attention_score = softmax(attention_score, dimension=-1)
    return einsum(attention_score, v, "b ... q_len k_len, b ... k_len d_v -> b ... q_len d_v")

