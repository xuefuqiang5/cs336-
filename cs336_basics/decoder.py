import torch
import torch.nn.functional as F
from einops import rearrange
import argparse
import os
from pathlib import Path
import yaml
from cs336_basics.Tokenizer import BPETokenizer
from cs336_basics.Transformer import TransformerLm

def get_next_token(logits, temperature, p):
    """
    基于 Top-P (Nucleus) Sampling 获取下一个 token
    :param logits: [batch_size, seq_length, vocab_size]
    :param temperature: 温度系数
    :param p: Top-P 累积概率阈值 (0.0 - 1.0)
    """
    # 通常生成任务只需要最后一个时间步的 logits
    # shape: [batch_size, vocab_size]
    logits = logits[:, -1, :] 
    
    # 处理 temperature 维度 (假设输入是 tensor)
    if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
        temperature = rearrange(temperature, "v -> 1 v")
    
    # 1. 计算概率分布
    probs = F.softmax(logits / temperature, dim=-1)
    
    # 2. 排序 (Sort)
    # torch.sort 返回 (values, indices)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 3. 计算累积概率 (Cumsum)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 4. 生成掩码 (Masking)
    # 移除累积概率超过阈值 p 的 token
    sorted_indices_to_remove = cumulative_probs > p
    
    # 技巧：将掩码向右移动一位，以保留第一个超过阈值的 token
    # (确保至少有一个 token 被选中，防止 p 很小时所有都被 mask 掉)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 5. 映射回原始索引 (Scatter)
    # 将排序后的掩码映射回原始词表顺序
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    # 6. 应用掩码并重新归一化
    # 将被移除的 token 概率置为 0
    probs = probs.masked_fill(indices_to_remove, 0.0)
    
    # 重新归一化概率分布 (因为去掉了一部分尾部概率)
    # 加上 1e-8 防止除以零
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    # 7. 采样 (Sampling)
    # shape: [batch_size, 1]
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint_path(explicit_path: str | None, checkpoint_dir: str) -> Path:
    if explicit_path:
        checkpoint_path = Path(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")
        return checkpoint_path

    checkpoint_root = Path(checkpoint_dir)
    candidates = sorted(
        checkpoint_root.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"在 {checkpoint_root} 下未找到 checkpoint_*.pt")
    return candidates[-1]


def generate_text(model, tokenizer, prompt, temperature, p, max_length=100):
    device = next(model.parameters()).device
    context_length = model.blocks[0].attention.max_seq_len
    ids_list = tokenizer.encode(prompt)
    ids = torch.tensor([ids_list], dtype=torch.long).to(device) # shape: [1, seq_len]
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    buffer = []

    model.eval()
    with torch.no_grad():
        for _ in range(max_length): # 使用 for 循环防止无限死循环
            ids_cond = ids[:, -context_length:]
            logits = model(ids_cond)
            next_token = get_next_token(logits, temperature, p)
            ids = torch.cat([ids, next_token], dim=-1)

            token_id = next_token.item()

            if token_id == eos_id:
                break
            word = tokenizer.decode([token_id])

            if "<|endoftext|>" in word:
                break

            print(word, end="", flush=True) # flush 确保实时打印
            buffer.append(word)

    return "".join(buffer) # 通常返回拼接后的完整字符串比返回 list 更好

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cs336_basics/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="I can still remember")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = BPETokenizer.from_files(
        config["vocab_filepath"],
        config["merges_filepath"],
        ["<|endoftext|>"],
    )
    model = TransformerLm(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        theta=config["theta"],
    ).to(device)

    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint,
        config.get("checkpoint_dir", "./checkpoints"),
    )
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    msg = model.load_state_dict(new_state_dict, strict=True)
    print(f"Model loaded successfully! {msg}")

    print(args.prompt, end=" ")
    generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        temperature=args.temperature,
        p=args.top_p,
        max_length=args.max_length,
    )
