import torch 
import argparse
import numpy as np
import os
import time
from tqdm import trange
from .bpe_train import bpe_train
from .Tokenizer import BPETokenizer, split2chunks
from .Transformer import TransformerLm
from .AdamW import AdamW
from .data_loading import data_loading
from .cross_entropy_loss import cross_entropy_loss
from .gradient_clipping import gradient_clipping

def init_args(): 
    parser = argparse.ArgumentParser()

    # ===================== tokenizer =====================
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_filepath", type=str, required=True)
    parser.add_argument("--merges_filepath", type=str, required=True)

    # ===================== model =====================
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--theta", type=float, default=10000.0)

    # ===================== optimizer =====================
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)


    # ===================== train =====================
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="int32")
    parser.add_argument("--log_interval", type=int, default=100)

    return parser.parse_args()

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
    return BPETokenizer.from_files(
        args.vocab_filepath, 
        args.merges_filepath,
        special_tokens
    ) 

   
def get_batch_iterable(args, tokenizer, endoftext="<|endoftext|>"):
    """
    生成器：循环处理 chunks 并产出多个 batches
    """
    # 假设 split2chunks 返回一个 chunk 生成器
    chunks = split2chunks(args.data_path, endoftext, args.buffer_size)
    
    for chunk in chunks:
        # 1. 编码当前 chunk
        id_list = tokenizer.encode(chunk)
        
        # 2. 如果当前 chunk 编码后太短，跳过
        if len(id_list) <= args.context_length:
            continue
            
        # 3. 计算这个 chunk 应该产出多少个 batch
        # 策略：为了让每个 token 都有机会被看到，产出 (chunk_len / context_length) * 某个倍数
        # 或者根据你的训练需求设置固定步数
        num_batches_per_chunk = max(1, len(id_list) // (args.context_length * 2))
        
        for _ in range(num_batches_per_chunk):
            batch = data_loading(id_list, args.batch_size, args.context_length, args.device)
            if batch is not None:
                yield batch # 返回 (x, y)

class DataLoader:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __iter__(self):
        # 每次调用 iter(obj) 时（比如进入 for 循环时），都会执行这里的代码
        return get_batch_iterable(self.args, self.tokenizer)

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
    for batch_idx, (x, y) in enumerate(data_loader):
        # 1. 数据移动到设备 (GPU/CPU)
        # 如果 get_batch_iterable 内部已经移动过，这里可以省略
        x, y = x.to(args.device), y.to(args.device)

        # 2. 前向传播
        # 假设模型输出为 logits, 形状为 (batch, seq_len, vocab_size)
        logits = model(x)
        
        # 3. 计算损失 (Cross Entropy)
        # 注意：CrossEntropyLoss 需要将 logits 展平为 (batch * seq_len, vocab_size)
        # y 展平为 (batch * seq_len)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # 4. 梯度累积处理：缩放损失
        loss = loss / grad_accum_steps
        loss.backward()

        # 5. 更新参数
        if (batch_idx + 1) % grad_accum_steps == 0:
            # 梯度裁剪 (防止梯度爆炸)
            if hasattr(args, "grad_clip"):
                gradient_clipping(model.parameters(), args.grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()

        # 6. 日志记录
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
    model = get_model(args)
    optimizer = AdamW(
        model.parameters, 
        args.lr, 
        args.weight_decay, 
        args.betas, 
        args.eps
    )
    tokenizer = get_tokenizer(args)
    data_loader = DataLoader(args, tokenizer)
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch} Start ---")
        epoch_loss = train(args, data_loader, model, optimizer)
        print(f"End of Epoch {epoch} | Average Loss: {epoch_loss:.4f}")

if __name__ == "__main__": 
    main()