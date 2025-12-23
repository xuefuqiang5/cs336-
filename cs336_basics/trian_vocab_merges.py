import json
import os
import base64
from .bpe_train import bpe_train

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
    vocab, merges = bpe_train(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # ---------- save vocab.json ----------
    vocab_path = os.path.join(output_dir, "vocab.json")

    vocab_json = {
        str(k): base64.b64encode(v).decode("utf-8")
        for k, v in vocab.items()
    }

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    # ---------- save merges.txt ----------
    merges_path = os.path.join(output_dir, "merges.txt")

    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', errors='replace')} "
                    f"{b.decode('utf-8', errors='replace')}\n")

    return vocab, merges

if __name__ == "__main__":

    input_path = "/Users/xuewenqi/code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    ouput_path = "./ouput"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    get_vocab_and_merges(input_path, ouput_path, vocab_size, special_tokens)