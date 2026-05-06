import numpy as np
from cs336_basics.Tokenizer import BPETokenizer

# ===================== 配置 =====================
bin_path = "data/data.bin"
vocab_path = "cs336_basics/Tokenizer/vocab.json"
merges_path = "cs336_basics/Tokenizer/merges.txt"
special_tokens = ["<|endoftext|>"]

num_tokens_to_read = 200
dtype = np.int32  # ⚠️ 和你生成 bin 时保持一致

# ===================== tokenizer =====================
tokenizer = BPETokenizer.from_files(
    vocab_path,
    merges_path,
    special_tokens
)

# ===================== 流式读取 =====================
tokens = np.memmap(
    bin_path,
    dtype=dtype,
    mode="r"
)

head_tokens = tokens[:num_tokens_to_read].tolist()

print("Token IDs:")
print(head_tokens)

# ===================== decode =====================
text = tokenizer.decode(head_tokens)

print("\nDecoded text:")
print("=" * 40)
print(text)
print("=" * 40)