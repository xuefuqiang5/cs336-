import os
import regex as re
from collections import Counter
from itertools import chain
import time
import heapq
def bpe_merge(
        byte_dict: dict[tuple, int], 
        best_pair: tuple[bytes, bytes],
        count: Counter
) -> dict[tuple, int]: 
    
    merged_dict = {}
    for word, freq in byte_dict.items(): 

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

        merged_dict[tuple(merged_tokens)] = freq
        
    return merged_dict 

def bpe_tokenizer(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start = time.time()
    with open(input_path, "r", encoding='utf-8') as f: 
        data = f.read()
    
    pat = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    data = re.split(pat, data)

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    data = list(chain.from_iterable(
        re.findall(pat, t) if t not in special_tokens else [t] for t in data
    ))

    pre_train_freq = Counter(x for x in data if x not in special_tokens)

    pre_train_freq = {word.encode('utf-8'): freq for word, freq in pre_train_freq.items()}

    vocab = {i: bytes([i]) for i in range(256)}
    next_vocab_key = 256
    existing_tokens = {bytes([i]) for i in range(256)}
     
    special_token_set = {special_token.encode('utf-8') for special_token in special_tokens} 
    for st in special_token_set: 
        vocab[next_vocab_key] = st
        existing_tokens.add(st)
        next_vocab_key += 1
    

    byte_freq = {
        tuple(bytes([b]) for b in word): freq
        for word, freq in pre_train_freq.items()
    } 
    end = time.time()
    print(f"pre-token cost {(end - start)}")

    epochs = vocab_size - (len(existing_tokens))

    merged_dict = byte_freq
    count = Counter(
            (word[i], word[i+1])
            for word, freq in byte_freq.items()
            for i in range(len(word) - 1)
            for _ in range(freq)
    )
    
    # best_pair = max(
    #     count,
    #     key=lambda x: (count[x], -ord(str(x[0])[0]))
    # )


    merges = []

    start = time.time()
    for _ in range(epochs): 
        max_freq = max([freq for freq in count.values()])
        candidates = [pair for pair, freq in count.items() if freq == max_freq]
        best_pair = max(candidates)
        merges.append(best_pair)
        count[best_pair] = 0
        merged_dict = bpe_merge(merged_dict, best_pair, count)
        # best_pair = max(
        #     count,
        #     key=lambda x: (count[x], -ord(str(x[0])[0]))
        # )
        

    end = time.time() 
    print(f"the merge process cost {end - start}")
    print(f"the vocab's size = {len(vocab.items())}, the merge' size = {len(merges)}") 
    for pair in merges: 
        new_token = pair[0] + pair[1]
        vocab[next_vocab_key] = new_token
        existing_tokens.add(new_token)
        next_vocab_key += 1

    return vocab, merges 
    


