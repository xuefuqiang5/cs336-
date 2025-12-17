import torch
import numpy as np

def data_loading(data, batch_size, context_length, device): 
    max_start = len(data) - context_length 
    starts = np.random.randint(0, max_start, size=batch_size)
    x, y = [], []
    for i in starts: 
        x.append(data[i: context_length+i])
        y.append(data[i+1: context_length+i+1])
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y
    