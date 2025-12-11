import torch
from torch import nn
from torch.nn.init import trunc_normal_
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        factory_kwargs = {}
        if device: 
            factory_kwargs['device'] = device
        if dtype: 
            factory_kwargs['dtype'] = dtype
        self.W = nn.Parameter(torch.randn(out_features, in_features, **factory_kwargs))
        std = math.sqrt(2.0 / (in_features + out_features))
        trunc_normal_(self.W, 0, std, -3*std, 3*std)
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        return super().load_state_dict(state_dict, strict, assign)
            
    def forward(
            self, 
            x: torch.Tensor
    ) -> torch.Tensor:

        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")