import torch 
from torch import nn
from .RMSNorm import RMSNorm
from .MultiheadSelfAttention import MultiheadSelfAttention
from .PositionwiseFeedforward import PositionwiseFeedforward
from .Embedding import Embedding
from .Linear import Linear
from .MultiheadSelfAttention import softmax

class TransformerBlock(nn.Module): 
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_ff: int,
            max_seq_len: int, 
            theta: float
    ):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len, True, theta)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.attention(self.rmsnorm1(x)) + x
        return self.ffn(self.rmsnorm2(output)) + output
    

class TransformerLm(nn.Module): 
    def __init__(
            self, 
            vocab_size: int, 
            context_length: int, 
            num_layers: int,
            d_model: int, 
            num_heads: int,
            d_ff: int, 
            theta: float
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers))
        self.rmsnorm = RMSNorm(d_model)
        self.output = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        o = self.embedding(x)
        for l in self.blocks: 
            o = l(o)
        o = self.rmsnorm(o)
        return self.output(o)
