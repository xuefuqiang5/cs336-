import torch
from torch import nn
from torch.nn.init import trunc_normal_
from einops import rearrange, einsum
from .Linear import Linear
import math

class PositionwiseFeedforward(nn.Module): 
    def __init__(
            self, 
            d_model: int, 
            d_ff: int,
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
    ):
        super().__init__()

        # d_ff = round(8/3 * d_model)
        # self.W_1 = nn.Parameter(torch.ones(d_ff, d_model, **factory_kwargs))
        # self.W_3 = nn.Parameter(torch.ones(d_ff, d_model, **factory_kwargs))
        # self.W_2 = nn.Parameter(torch.ones(d_model, d_ff, **factory_kwargs))
        self.W1 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)
        self.W3 = Linear(d_model, d_ff, device, dtype)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        W1x = self.W1(x)
        silu = W1x * torch.sigmoid(W1x)
        # W3x = einsum(self.W_3, x, "d_ff d_model, ... d_model -> ... d_ff")
        W3x = self.W3(x)
        # return einsum(self.W_2, silu*W3x, "d_model d_ff, ... d_ff -> ... d_model")
        return self.W2(silu*W3x)

