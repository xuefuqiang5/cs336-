import torch 
import argparse
import numpy as np
import os

from tqdm import trange
from .bpe_train import bpe_train
from .Tokenizer import BPETokenizer, split2chunks
from .Transformer import TransformerLm
from .AdamW import AdamW
from .data_loading import data_loading
from .cross_entropy_loss import cross_entropy_loss
from .gradient_clipping import gradient_clipping

def train():
    parser = argparse.ArgumentParser()

    # ===================== 数据相关 =====================
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--buffer_size", type=int, default=1024 * 1024)

    # ===================== tokenizer =====================
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--special_tokens", nargs="+", required=True)

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


    # ===================== data / batching =====================
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)

    # ===================== 其他 =====================
    parser.add_argument("--dtype", type=str, default="int32")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    device = torch.device(args.device)

    model = TransformerLm(
        args.vocab_size, 
        args.context_length, 
        args.num_layers, 
        args.num_heads, 
        args.num_heads, 
        args.dff, 
        args.theta
    )
    optimizer = AdamW(
        model.parameters, 
        args.lr, 
        args.weight_decay, 
        args.betas, 
        args.eps
    )
    # =====================================================
    # 1️⃣ 文本 → token ids（流式）
    # =====================================================
    if not os.path.exists(args.out_path):
        print("Tokenizing dataset (streaming)...")

        tokenizer = BPETokenizer.from_files(
            args.vocab_path,
            args.merges_path,
            args.special_tokens,
        )

        chunks = split2chunks(
            args.data_path,
            args.special_tokens[0],
            buffer_size=args.buffer_size,
        )

        with open(args.out_path, "wb") as f:
            for chunk in chunks:
                token_ids = tokenizer.encode(chunk)
                np.asarray(token_ids, dtype=args.dtype).tofile(f)

    # =====================================================
    # 2️⃣ memory-mapped dataset
    # =====================================================
    data = np.memmap(
        args.out_path,
        dtype=args.dtype,
        mode="r",
    )

    num_tokens = len(data)
    steps_per_epoch = num_tokens // (args.batch_size * args.context_length)

    print(f"Total tokens: {num_tokens}")
    print(f"Steps per epoch: {steps_per_epoch}")

    model.to(device)
    model.train()

    # =====================================================
    # 3️⃣ Training loop
    # =====================================================
    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0

        for step in trange(steps_per_epoch, desc=f"Epoch {epoch}"):
            # ---- sample batch ----
            x, y = data_loading(
                data,
                args.batch_size,
                args.context_length,
                device,
            )

            # ---- forward ----
            logits = model(x)  # (B, L, vocab_size)

            # ---- loss ----
            loss = cross_entropy_loss(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            # ---- backward ----
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_clipping(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

        avg_loss = epoch_loss / steps_per_epoch
        print(f"[Epoch {epoch}] avg loss = {avg_loss:.4f}")