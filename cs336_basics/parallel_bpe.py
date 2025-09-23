import os
import regex as re
from collections import Counter
from itertools import chain
from multiprocessing import Pool, cpu_count
from functools import partial
import heapq

def pre_tokenization(text: str, special_tokens: list[str]) -> tuple[dict[tuple[bytes], int], Counter]:

    pat = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    text = re.split(pat, text)

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text = list(chain.from_iterable(
        re.findall(pat, t) if t not in special_tokens else [t] for t in text
    ))

    pre_train_freq = Counter(x for x in text if x not in special_tokens)

    pre_train_freq = {word.encode('utf-8'): freq for word, freq in pre_train_freq.items()}

    count = Counter(
        (word[i], word[i+1])
        for word, freq in pre_train_freq.items()
        for i in range(len(word) - 1)
        for _ in range(freq)
    )

    return pre_train_freq, count

     
def split_text_to_chunk(input_paht: str, work_number: int) -> None:  
    raise NotImplemented

def merge_update_heap(
        bytes_dict: dict[tuple, int], 
        best_pair: tuple, 
        count: dict[tuple, int]
)-> dict[tuple, int]: 

    new_bytes_dict = {}

    for word, freq in bytes_dict.items(): 

            merged_tokens, i = [], 0

            while i < len(word): 
                if (i < len(word) - 1) and ((word[i], word[i+1]) == best_pair): 
                    # 当word[i] 是之前pair 的右元素
                    # 当new_pair 的右元素为best_pair
                    if i > 0: 
                        count[(word[i-1], word[i])] -= freq
                        count[(word[i-1], best_pair[0] + best_pair[1])] += freq

                    # 当word[i+1] 为之前pair 的左元素
                    # 当new_pair 的左元素为best_pair
                    if i + 2 < len(word): 
                        count[(word[i+1], word[i+2])] -= freq
                        count[(best_pair[0] + best_pair[1], word[i+2])] += freq 
                    merged_tokens.append(word[i] + word[i+1])
                    i += 2

                else: 
                    merged_tokens.append(word[i])
                    i += 1

            new_bytes_dict[tuple(merged_tokens)] = freq
    
    return new_bytes_dict 

def parallel_bpe_train(
       input_path: str, 
       vocab_size: int, 
       special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    work_number = cpu_count()
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

    bytes_tokenize_with_special_token = partial(bytes_tokenize, special_tokens=special_tokens)
    with Pool(processes=work_number) as pool: 
            bytes_dicts = pool.map(bytes_tokenize_with_special_token, parts)

    bytes_dicts = {}
    counts = Counter(
            (word[i], word[i+1])
            for sub_bytes_dict in bytes_dicts
            for word, freq in sub_bytes_dict.items()
            for i in range(len(word) - 1)
            for _ in range(freq)
        )    

       

    heap = [(-freq, pair) for pair, freq in counts] 
    heapq.heapify(heap)

    merges = []
    with Pool(processes=work_number) as pool: 
        for _ in range(epochs): 
            bytes_pairs_count = pool.map(get_counter, bytes_dicts)

            bytes_pairs_count = sum(bytes_pairs_count, Counter())
            max_freq = max(bytes_pairs_count.values())
            candidates = [pair for pair, freq in bytes_pairs_count.items() if freq == max_freq]
            best_pair = max(candidates)
            merges.append(best_pair)

            best_pair = None
            merge_with_best_pair = partial(merge_update_heap, best_pair=best_pair)

            bytes_dicts = pool.map(merge_with_best_pair, bytes_dicts)

    print(f"the vocab's size = {len(vocab.items())}, the merge' size = {len(merges)}") 
    for pair in merges: 
        new_token = pair[0] + pair[1]
        vocab[next_vocab_key] = new_token
        existing_tokens.add(new_token)
        next_vocab_key += 1

    return vocab, merges

   # 1. 进行持久化处理 