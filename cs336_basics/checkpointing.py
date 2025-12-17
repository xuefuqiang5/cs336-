import torch
from typing import Union, BinaryIO
import os

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: Union[str, os.PathLike, BinaryIO]):
    """
    Save model, optimizer, and iteration into a checkpoint.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    """
    Load checkpoint from src and restore model, optimizer states.
    Returns the saved iteration number.
    """
    checkpoint = torch.load(src, map_location="cpu")  # map_location 避免 cuda/cpu mismatch
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]
    return iteration