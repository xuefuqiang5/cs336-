import torch 
from torch import nn
from einops import einsum, rearrange
from torch.nn.init import trunc_normal_
from .RoPE import RoPE
from .Linear import Linear
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

# class MultiheadSelfAttention(nn.Module): 
#     def __init__(
#             self,
#             d_model: int, 
#             num_heads: int,
#             max_seq_length: int | None = None, 
#     ):
#         super().__init__()
#         self.d_k = d_model // num_heads
#         self.W_q = nn.Parameter(torch.Tensor(torch.randn(d_model, d_model)))
#         self.W_k = nn.Parameter(torch.Tensor(torch.randn(d_model, d_model)))
#         self.W_v = nn.Parameter(torch.Tensor(torch.randn(d_model, d_model)))
#         self.W_o = nn.Parameter(torch.Tensor(torch.randn(d_model, d_model)))
#         self.max_seq_length = max_seq_length

#         std = math.sqrt(2.0 / (d_model + d_model))
#         trunc_normal_(self.W_q, 0, std, -3*std, 3*std)
#         trunc_normal_(self.W_k, 0, std, -3*std, 3*std)
#         trunc_normal_(self.W_v, 0, std, -3*std, 3*std)
#         trunc_normal_(self.W_o, 0, std, -3*std, 3*std)

#     def forward(self, x:  torch.Tensor) -> torch.Tensor: 

#         q = einsum(x, self.W_q, "... l d_model, d_model d_model -> ... l d_model")
#         k = einsum(x, self.W_k, "... l d_model, d_model d_model -> ... l d_model")
#         v = einsum(x, self.W_v, "... l d_model, d_model d_model -> ... l d_model")

#         q = rearrange(q, "... l (h d_k) -> ... h l d_k", d_k=self.d_k)
#         k = rearrange(k, "... l (h d_k) -> ... h l d_k", d_k=self.d_k)
#         v = rearrange(v, "... l (h d_k) -> ... h l d_k", d_k = self.d_k)

#         if self.max_seq_length: 
#             rope = RoPE(theta=-1e5, d_k=self.d_k, max_seq_len=self.max_seq_length)
#             q = rope(q)
#             k = rope(k)
#         L = x.size(-2)
#         mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
#         multihead_attention = scaled_dot_product_attention(q, k, v, mask)
#         multihead_attention = rearrange(multihead_attention, "... h l d_k -> ... l (h d_k)")
#         return einsum(multihead_attention, self.W_o, "... l d_model, d_model d_model -> ... l d_model")

class MultiheadSelfAttention(nn.Module): 
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            max_seq_len: int | None = None, 
            use_rope: bool = False,
            theta: float | None = None,
            token_positions: torch.Tensor | None = None, 
    ):
        super().__init__()
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.theta = theta
        self.token_positions = token_positions
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # qkv.shape = [3*d_model, d_model]
        qkv = torch.cat([self.W_q.W, self.W_k.W, self.W_v.W])
        # qkv_output.shape = [d_model, 3*d_model]
        qkv_output = x @ qkv.T
        q, k, v = qkv_output.chunk(3, dim=-1)
        q = rearrange(q, "... l (h d_k) -> ... h l d_k", h=self.num_heads)
        k = rearrange(k, "... l (h d_k) -> ... h l d_k", h=self.num_heads)
        v = rearrange(v, "... l (h d_k) -> ... h l d_k", h=self.num_heads)
        if self.use_rope: 
            rope = RoPE(self.theta, self.d_k, max_seq_len=self.max_seq_len)
            q = rope(q, self.token_positions)
            k = rope(k, self.token_positions)
        
        L = x.size(-2)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        mask = mask[None, None, :, :]
        attention_socres = scaled_dot_product_attention(q, k, v, ~mask)
        attention_socres = rearrange(attention_socres, "... h l d_k -> ... l (h d_k)", h=self.num_heads)
        return self.W_o(attention_socres)



