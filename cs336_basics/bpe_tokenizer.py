import os
import regex as re
from collections import Counter
from itertools import chain

def merge(byte_dict: dict[tuple, int], merge: list[tuple]): 
    pair_dict = Counter(
        (word[i], word[1+1])
        for word, freq in byte_dict.items()
        for i in range(len(word) - 1)
        for _ in range(freq)
        for pair in [(word[i] + word[i+1])]
    )    

    max_freq = max(pair_dict.values())
    candidates = [pair for pair, freq in pair_dict.items() if freq == max_freq]
    best_tuple = max(candidates, key=lambda p: p[0] + p[1])
    best_pair = best_tuple[0] + best_tuple[1]
    merge.append(best_tuple[0], best_tuple[1])

    merged_dict = {}
    for word, freq in byte_dict.items(): 
        merged_tokens, i = [], 0

        while i < len(word): 
            if (i < len(word) - 1) and word[i] + word[i+1] == best_pair: 
                merged_tokens.append(best_pair)
                i += 2
            else: 
                merged_tokens.append(word[i])
                i += 1

        merged_dict[tuple(merged_tokens)] = freq
           
    return merged_dict 

def bpe_tokenizer(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding='utf-8') as f: 
        data = f.read()
    
    pat = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    data = re.split(pat, data)

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    data = list(chain.from_iterable(
        re.findall(pat, t) if t not in special_tokens else [t] for t in data
    ))
    pre_train_freq = Counter(data)
    [print(word) for idx, word in enumerate(pre_train_freq.items()) if idx < 3]

    byte_freq = {
        (tuple(word.encode("utf-8")) if word not in special_tokens else word): freq
        for word, freq in pre_train_freq.items()
    }
    print(type(byte_freq))
    [print(word) for idx, word in enumerate(byte_freq.items()) if idx < 3]

    init_vocab_size = [len(word) for word, _ in byte_freq.items()]
    init_vocab_size = sum(init_vocab_size)

    epochs = vocab_size - (init_vocab_size + len(special_tokens))
    merged_dict = byte_freq

    print(f"the epochs's value is {epochs}")
    
    merge = [()]

    for _ in range(epochs): 
        merged_dict = merge(merged_dict)
    
    
    unique_tokens = {token for tokens in merged_dict.key() for token in tokens} 
    vocab = {idx: token for idx, token in enumerate(unique_tokens)}

    return vocab, merge
    


    

data_path = "/Users/xuewenqi/code/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
vocab_size = 1000
special_tokens = [
    "<|pad|>",        # padding，用于对齐 batch
    "<|unk|>",        # unknown，未知 token
    "<|bos|>",        # beginning of sequence，序列开始
    "<|eos|>",        # end of sequence，序列结束
    "<|sep|>",        # separator，分隔符
    "<|cls|>",        # classification token（某些任务用）
    "<|mask|>",       # mask，用于掩码语言模型
    "<|endoftext|>",  # GPT 风格的文档结束 token
]

vocab, merge = bpe_tokenizer(data_path, vocab_size, special_tokens)
print(type(vocab), type(merge))
