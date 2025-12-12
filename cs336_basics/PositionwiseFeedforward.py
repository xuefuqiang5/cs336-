import torch
from torch import nn
from torch.nn.init import trunc_normal_
from einops import rearrange, einsum
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
        factory_kwargs = {}
        if device: 
            factory_kwargs['device'] = device
        if dtype: 
            factory_kwargs['dtype'] = dtype
 
        # d_ff = round(8/3 * d_model)
        self.W_1 = nn.Parameter(torch.ones(d_ff, d_model, **factory_kwargs))
        self.W_3 = nn.Parameter(torch.ones(d_ff, d_model, **factory_kwargs))
        self.W_2 = nn.Parameter(torch.ones(d_model, d_ff, **factory_kwargs))

        std = math.sqrt(2.0 / (d_model + d_ff))
        trunc_normal_(self.W_1, 0, std, -3*std, 3*std) 
        trunc_normal_(self.W_2, 0, std, -3*std, 3*std) 
        trunc_normal_(self.W_3, 0, std, -3*std, 3*std) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        W1x = einsum(self.W_1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = W1x * torch.sigmoid(W1x)
        W3x = einsum(self.W_3, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.W_2, silu*W3x, "d_model d_ff, ... d_ff -> ... d_model")

