import json
import os
import base64
from cs336_basics.parallel_bpe import train_bep
from collections.abc import Iterator

def split2chunks(data_path: str, endoftext: str, buffer_size: int = 1024 * 1024) -> Iterator[str]: 
    buffer = ""
    with open(data_path, "r", encoding='utf-8') as f: 
        while True: 
            data = f.read(buffer_size)
            if not data: 
                break
            buffer += data
            while True:
                idx = buffer.find(endoftext)
                if idx == -1: 
                    break
                end = idx + len(endoftext)
                chunk = buffer[:end]
                yield chunk
                buffer = buffer[end:]
    if buffer: 
        buffer += endoftext
        yield buffer 
def bytes_to_unicode():
    """
    创建一个从 byte 到可打印 Unicode 字符的映射。
    """
    # 基础可打印字符范围
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    # 将不可见字符映射到不常用的 Unicode 区域
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 预实例化映射表
BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

def get_mapped_vocab(vocab, byte_encoder):
    # 如果你的 vocab 里的 key 是 bytes (如 b'hello')
    # 或者包含原始字符，我们需要将其转换为映射后的字符串
    new_vocab = {}
    for idx, token in vocab.items():
        # 将 token (假设是 bytes 类型) 转换为映射后的字符串
        if isinstance(token, bytes):
            new_token = "".join([byte_encoder[b] for b in token])
        else:
            # 如果是原始字符串，先转 byte 再映射
            new_token = "".join([byte_encoder[b] for b in token.encode('utf-8')])
        new_vocab[idx] = new_token
    return new_vocab

def get_mapped_merges(merges, byte_encoder):
    """
    merges: [(bytes, bytes)] 
    返回值: [(str, str)]
    """
    new_merges = []
    for p1, p2 in merges:
        # 将 pair 中的每一项从 bytes 转换为 mapped string
        m1 = "".join([byte_encoder[b] for b in p1])
        m2 = "".join([byte_encoder[b] for b in p2])
        new_merges.append((m1, m2))
    return new_merges
    
def save_bpe(vocab, merges, output_dir):
    """
    在指定目录下创建并保存 vocab.json 和 merges.txt
    """
    # 1. 如果目录不存在，则创建目录
    if not os.path.exists(output_dir):
        print(f"no: {output_dir}")

    # 2. 定义文件完整路径
    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")

    # 3. 保存 Vocab (JSON 格式)
    with open(vocab_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 保证不可见字符映射后的 Unicode 字符（如 Ġ）能原样保存
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    
    # 4. 保存 Merges (文本格式)
    with open(merges_path, "w", encoding="utf-8") as f:
        # 写入版本信息（参考 GPT-2/RoBERTa 规范）
        f.write("#version: 0.2\n")
        for pair in merges:
            # pair 是元组，如 ('Ġ', 't')，保存为 "Ġ t"
            f.write(f"{pair[0]} {pair[1]}\n")

    print(f"Vocab 已保存至: {vocab_path}")
    print(f"Merges 已保存至: {merges_path}")
def get_vocab_and_merges(
    input_path: str,
    output_dir: str,
    vocab_size: int,
    special_tokens: list[str]
):
    """
    Train BPE and save vocab.json and merges.txt.

    Args:
        input_path (str): path to input training text
        output_dir (str): directory to save vocab and merges
        vocab_size (int): target vocab size
        special_tokens (list[str]): list of special tokens

    Returns:
        vocab (dict[int, bytes])
        merges (list[tuple[bytes, bytes]])
    """
    os.makedirs(output_dir, exist_ok=True)

    # train
    vocab, merges = train_bep(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    mapped_vocab = get_mapped_vocab(vocab, BYTE_ENCODER)
    mapped_merges = get_mapped_merges(merges, BYTE_ENCODER)
    save_bpe(mapped_vocab, mapped_merges, output_dir)
if __name__ == "__main__":

    input_path = "data/data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "cs336_basics/Tokenizer"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    get_vocab_and_merges(input_path, output_dir, vocab_size, special_tokens)