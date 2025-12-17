import torch
from collections.abc import Callable 
from typing import Optional
from torch import optim, nn
import math

class AdamW(optim.Optimizer): 
    def __init__(
            self, 
            params,
            lr, 
            weight_decay, 
            betas, 
            eps,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay, 
            "eps": eps
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure == None else closure()
        for group in self.param_groups: 
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]: 
                if p.grad == None: 
                    continue
                state = self.state[p]
                if len(state) == 0:  
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m = state["m"]
                v = state["v"]
                t = state["step"] + 1
                state["step"] = t
                g = p.grad
                m.mul_(beta1).add_(g, alpha=1-beta1)
                v.mul_(beta2).addcmul_(g, g, value=1-beta2)
                beta1_hat = 1 - beta1 ** t
                beta2_hat = 1 - beta2 ** t
                lr_t = lr * math.sqrt(beta2_hat) / beta1_hat
                denom = v.sqrt().add_(eps)
                p.add_(p, alpha=-lr * wd)
                p.addcdiv_(m, denom, value=-lr_t)

        return loss   
