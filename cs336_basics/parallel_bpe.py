import os
import regex as re
from collections import Counter
from itertools import chain
from multiprocessing import Pool, cpu_count

def bytes_tokenize(
        data: str,
        special_tokens: list[str], 
) -> dict[tuple, int]: 
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    data = list(chain.from_iterable(
        re.findall(pat, t) if t not in special_tokens else [t] for t in data
    )) 

    pre_train_freq = Counter(x for x in data if x not in special_tokens)
    pre_train_freq = {word.encode('utf-8'): freq for word, freq in pre_train_freq.items()}

    byte_freq = {
            tuple(bytes([b]) for b in word): freq
            for word, freq in pre_train_freq.items()
        }

    return byte_freq

def merge(bytes_dict: dict[tuple, int], best_pair: tuple)-> dict[tuple, int]: 

    new_bytes_dict = {}
    for word, _ in bytes_dict.items(): 
            merged_tokens, i = [], 0

            while i < len(word): 
                if (i < len(word) - 1) and ((word[i], word[i+1]) == best_pair): 
                    merged_tokens.append(word[i] + word[i+1])

                    i += 2
                else: 
                    merged_tokens.append(word[i])
                    i += 1

    return new_bytes_dict 

def get_counter(sub_bytes_dict: dict[tuple, int]): 
    pair_dict = Counter(
            (word[i], word[i+1])
            for word, freq in sub_bytes_dict.items()
            for i in range(len(word) - 1)
            for _ in range(freq)
        )    
    
    return pair_dict

def parallel_bpe_train(
       input_path: str, 
       vocab_size: int, 
       special_tokens: list[str],
       work_number: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path, mode="r", encoding='utf-8') as f:
        data = f.read()
    
    parts = data.split("<|endoftext|>")
    work_number = len(parts) if len(parts) < work_number else work_number

    vocab = {i: bytes([i]) for i in range(256)}
    next_vocab_key = 256
    existing_tokens = {bytes([i]) for i in range(256)}

    special_token_set = {special_token.encode('utf-8') for special_token in special_tokens} 
    for st in special_token_set: 
        vocab[next_vocab_key] = st
        existing_tokens.add(st)
        next_vocab_key += 1

    epochs = vocab_size - (len(existing_tokens))




    