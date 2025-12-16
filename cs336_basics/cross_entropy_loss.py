from .MultiheadSelfAttention import softmax
import torch
from einops import einsum, reduce
def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor): 
    '''
    logits.shape = [batch_size, ..., vocab_size] 
    target.shape = [batch_size, ..., 1]
    '''

    logsumexp = torch.logsumexp(logits, dim=-1)
    diff = torch.gather(logits, -1, target.unsqueeze(-1)).squeeze(-1)
    # l.shape = [batch_size, seq_length]
    l = logsumexp - diff

    return l.mean()
    # return torch.sum(l, dim=0) / l.size(0)
