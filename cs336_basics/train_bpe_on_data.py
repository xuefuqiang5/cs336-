from cs336_basics.parallel_bpe import train_bep
import json
import os
def save_vocab(vocab: dict[int, bytes], path: str):
    vocab_json = {
        str(i): v.decode("utf-8", errors="ignore")
        for i, v in vocab.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)


def save_merges(merges: list[tuple[bytes, bytes]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            a_str = a.decode("utf-8", errors="ignore")
            b_str = b.decode("utf-8", errors="ignore")
            f.write(f"{a_str} {b_str}\n")

if __name__ == "__main__":

    data_path = "/Users/xuewenqi/code/cs336/cs336-/data/data/TinyStoriesV2-GPT4-train.txt"
    output_path = "/Users/xuewenqi/code/cs336/cs336-/cs336_basics/Tokenizer"

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    os.makedirs(output_path, exist_ok=True)

    print("🚀 Start training BPE...")
    vocab, merges = train_bep(
        input_path=data_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    vocab_path = os.path.join(output_path, "vocab.json")
    merges_path = os.path.join(output_path, "merges.txt")

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    print("✅ BPE training finished")
    print(f"Vocab saved to  : {vocab_path}")
    print(f"Merges saved to : {merges_path}")
