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
from cs336_basics.checkpointing import save_checkpoint
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


def evaluate_model(args, valid_data_loader, model):
    model.eval()  # 评估模式
    total_loss = 0.0
    total_tokens = 0  # 用于语言模型等 token 级 loss
    steps_per_epoch = valid_data_loader.get_len() // (args.batch_size * args.context_length)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(valid_data_loader):
            if batch_idx >= steps_per_epoch:
                break

            x, y = x.to(args.device), y.to(args.device)

            logits = model(x)

            loss = cross_entropy_loss(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            num_tokens = y.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    return {
        "val_loss": avg_loss,
    }
def train(args, train_data_loader, valid_data_loader, model, optimizer):
    model.train()  # 确保模型处于训练模式（开启 Dropout 等）
    total_loss = 0
    start_time = time.time()
    grad_accum_steps = getattr(args, "grad_accum_steps", 1)
    optimizer.zero_grad()
    # data_loader 是 get_batch_iterable 返回的生成器
    steps_per_epoch = train_data_loader.get_len() // (args.batch_size * args.context_length)
    for batch_idx, (x, y) in enumerate(tqdm(train_data_loader, mininterval=30.0)):
        if batch_idx >= steps_per_epoch:
            break
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
            print(
                f"Batch {batch_idx} | "
                f"Loss: {avg_loss:.4f} | "
                f"Ms/Batch: {elapsed * 1000 / (batch_idx + 1):.2f} | "
                f"Val loss: {evaluate_model(args, valid_data_loader, model)['val_loss']}"
            )
    return total_loss / (batch_idx + 1)

def main():
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    args = init_args()
    args.device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = get_model(args).to(args.device)
    from cs336_basics.debug_utils import DeviceDetective
    DeviceDetective.check_model_params(model)
    optimizer = AdamW(
        model.parameters(), 
        args.lr, 
        args.weight_decay, 
        args.betas, 
        float(args.eps)
    )
    train_data_loader = DataLoader("data/TinyStoriesV2-GPT4-train.bin", args.batch_size, args.context_length, args.device)
    valid_data_loader = DataLoader("data/TinyStoriesV2-GPT4-valid.bin", args.batch_size, args.context_length, args.device)
    train(args, train_data_loader, valid_data_loader, model, optimizer)

if __name__ == "__main__": 
    main()