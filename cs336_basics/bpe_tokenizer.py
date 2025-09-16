import os
import regex as re
from collections import Counter

def merge(byte_dict: dict[tuple: int]): 
    pair_dict = Counter(
        pair
        for word, freq in byte_freq.items()
        for i in range(len(word) - 1)
        for _ in range(freq)
        for pair in [(word[i] + word[i+1])]
    )    
    max_freq = max(pair_dict.values)
    candidates = [pair for pair, freq in pair_dict.items() if freq == max_freq]
    best_pair = max(candidates)
    merged_dict = {}
    # for word, freq in byte_dict.items(): 
    #     merged_tokens, i = [], 0

    #     while i < len(word): 
    #         if (i < len(word) - 1) and word[i] + word[i+1] == best_pair: 
    #             merged_tokens.append(best_pair)
    #             i += 2
    #         else: 
    #             merged_tokens.append(word[i])
    #             i += 1

    #     merged_dict[tuple(merged_tokens)] = freq
    for word, freq in byte_dict.items(): 
        it = iter(word)
        merged_tokens = []
        for ch in it: 
            try: 
                n = next(it)
                if ch + n == best_pair: 
                    merged_tokens.append(best_pair)
                else:
                    merged_tokens.append(ch)
                    merged_tokens.append(n)
            except StopIteration: 
                merged_tokens.append(ch)

        merged_dict[tuple(merged_tokens)] = freq
            
    return merged_dict 

def bpe_tokenizer(
        input_path: str, 
        vocab_size: int, 
        special_token: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding='utf-8') as f: 
        data = f.read()
    
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    data = re.findall(pat, data)
    pre_train_freq = Counter(data)
    byte_freq = {
        (tuple(word.encode("utf-8")) if word not in special_token else word): freq
        for word, freq in pre_train_freq.items()
    }
    init_vocab_size = [len(word) for word, _ in byte_freq]
    init_vocab_size = sum(init_vocab_size)

    epochs = vocab_size - (init_vocab_size + len(special_token))
    merged_dict = byte_freq

    for _ in range(epochs): 
        merged_dict = merge(merged_dict)


    

raw_data = {"hello": 1, "word": 2, "xihuan": 3}    
byte_freq = {
    tuple(word.encode('utf-8')): freq
    for word, freq in raw_data.items()
}

pair_dict = Counter(
    pair
    for word, freq in byte_freq.items()
    for i in range(len(word) - 1)
    for _ in range(freq)
    for pair in [(word[i] + word[i+1])]
)
print(pair_dict)

x = (1, 2, 3, 4)

