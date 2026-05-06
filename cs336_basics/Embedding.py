import torch
from torch import nn
from torch.nn.init import trunc_normal_

class Embedding(nn.Module): 
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int,
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        factory_kwargs = {}
        if device: 
            factory_kwargs['device'] = device
        if dtype: 
            factory_kwargs['dtype'] = dtype
 
        self.embedding = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, **factory_kwargs))
        trunc_normal_(self.embedding, 0, 1, -3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]