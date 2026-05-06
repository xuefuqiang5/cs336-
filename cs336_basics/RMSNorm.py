import torch 
from torch import nn
from torch.nn.init import trunc_normal_
from einops import reduce

class RMSNorm(nn.Module): 
    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-5, 
            device=None, 
            dtype=None
    ):
        super().__init__()
        factory_kwargs = {}
        if device: 
            factory_kwargs['device'] = device
        if dtype: 
            factory_kwargs['dtype'] = dtype
        self.g = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        rms_a = torch.sqrt(1/self.d_model * reduce(x**2, "... d -> ... 1", "sum") + self.eps)
        return x / rms_a * self.g
