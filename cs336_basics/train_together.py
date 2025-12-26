import torch 
import argparse
import numpy as np
import os
import time
from tqdm import trange
from tqdm import tqdm
from cs336_basics.bpe_train import bpe_train
from cs336_basics.Tokenizer import BPETokenizer, split2chunks
from cs336_basics.Transformer import TransformerLm
from cs336_basics.AdamW import AdamW
from cs336_basics.data_loading import data_loading
from cs336_basics.cross_entropy_loss import cross_entropy_loss
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.parallel_bpe import train_bep
from cs336_basics.DataLoader import DataLoader
import yaml
import argparse
import yaml
import os

class ConfigNamespace:
    """简单的类，用于将字典转换为可以使用 . 访问的对象"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def init_args():
    # 1. 预设 YAML 文件的默认路径（如果命令行也不传 --config，就用这个）
    DEFAULT_CONFIG_PATH = "/home/xuewenqi/cs336-/cs336_basics/config.yaml"

    # 2. 依然保留 argparse 仅用于获取 --config 路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config file")
    
    # 获取 config 文件的路径
    temp_args, _ = parser.parse_known_args()
    config_path = temp_args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}，请确保文件存在或通过 --config 指定。")

    # 3. 读取 YAML 文件
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 4. 定义默认值 (防止 YAML 中漏写参数导致后面代码崩溃)
    defaults = {
        "context_length": 1024,
        "num_layers": 12,
        "d_model": 768,
        "num_heads": 12,
        "d_ff": 3072,
        "theta": 10000.0,
        "lr": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "batch_size": 32,
        "epochs": 10,
        "grad_clip": 1.0,
        "dtype": "int32",
        "log_interval": 100,
        "device": "cuda" if os.path.exists("/dev/nvidia0") else "cpu" # 自动检测设备
    }

    # 5. 合并：用 YAML 的值覆盖默认值
    defaults.update(config_dict)

    # 6. 返回一个可以使用 args.xxx 访问的对象
    return ConfigNamespace(defaults)
def get_model(args):
    return TransformerLm(
        args.vocab_size, 
        args.context_length, 
        args.num_layers, 
        args.d_model,
        args.num_heads, 
        args.d_ff, 
        args.theta
    )
def get_tokenizer(args, special_tokens): 

    return BPETokenizer.from_files(args.vocab_filepath, args.merges_filepath, special_tokens)


def train(args, data_loader, model, optimizer):
    """
    训练函数：负责一个 Epoch 的数据迭代
    """
    model.train()  # 确保模型处于训练模式（开启 Dropout 等）
    
    total_loss = 0
    start_time = time.time()
    
    # 梯度累积步数，如果 args 中没定义，默认为 1
    grad_accum_steps = getattr(args, "grad_accum_steps", 1)
    
    # 清空初始梯度
    optimizer.zero_grad()

    # data_loader 是 get_batch_iterable 返回的生成器
    for batch_idx, (x, y) in enumerate(tqdm(data_loader)):
        x, y = x.to(args.device), y.to(args.device)

        logits = model(x)
        
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        
        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if hasattr(args, "grad_clip"):
                gradient_clipping(model.parameters(), args.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            # 计算当前的平均损失
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Batch {batch_idx} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Ms/Batch: {elapsed * 1000 / (batch_idx + 1):.2f}")

    return total_loss / (batch_idx + 1)

def main():
    args = init_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args).to(args.device)
    optimizer = AdamW(
        model.parameters(), 
        args.lr, 
        args.weight_decay, 
        args.betas, 
        args.eps
    )
    data_loader = DataLoader("data/data.bin", args.batch_size, args.context_length, args.device)
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch} Start ---")
        epoch_loss = train(args, data_loader, model, optimizer)
        print(f"End of Epoch {epoch} | Average Loss: {epoch_loss:.4f}")

if __name__ == "__main__": 
    main()