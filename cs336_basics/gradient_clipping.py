import torch
import math

@torch.no_grad()
def gradient_clipping(params, max_norm):
    """
    Clip gradients by global L2 norm.
    """
    # 1. 计算所有梯度的 L2 norm
    total_norm_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad
        total_norm_sq += grad.pow(2).sum().item()

    total_norm = math.sqrt(total_norm_sq)

    # 2. 如果不需要剪裁，直接返回
    if total_norm <= max_norm:
        return total_norm

    # 3. 计算缩放因子
    scale = max_norm / (total_norm + 1e-6)

    # 4. 对所有梯度做统一缩放
    for p in params:
        if p.grad is None:
            continue
        p.grad.mul_(scale)

    return total_norm