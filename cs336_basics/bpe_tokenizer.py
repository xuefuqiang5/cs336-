import os
import regex as re
from collections import Counter
from itertools import chain

def gpt2_bytes_to_unicode_local() -> dict[int, str]:

    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)) 
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

# input parameters 
def bpe_merge(
        byte_dict: dict[tuple, int], 
        merge: list[tuple],
) -> dict[tuple, int]: 
    pair_dict = Counter(
        (word[i], word[i+1])
        for word, freq in byte_dict.items()
        for i in range(len(word) - 1)
        for _ in range(freq)
        # for pair in [(word[i] + word[i+1])]
    )    
    
    # [print(word) for idx, word in enumerate(pair_dict.items()) if idx < 3]

    max_freq = max(pair_dict.values())
    candidates = [pair for pair, freq in pair_dict.items() if freq == max_freq]
    # print(f"the candidate pair is {candidates}")
    best_tuple = max(candidates)

    merge.append((best_tuple[0], best_tuple[1]))
    # print(f"the best tuple is {best_tuple}")

    merged_dict = {}
    for word, freq in byte_dict.items(): 
        merged_tokens, i = [], 0

        while i < len(word): 
            if (i < len(word) - 1) and ((word[i], word[i+1]) == best_tuple): 
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
    with open(input_path, "r", encoding='utf-8') as f: 
        data = f.read()
    
    pat = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    data = re.split(pat, data)

    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # split text to "{"word", "hello", "nihao"...} format"
    data = list(chain.from_iterable(
        re.findall(pat, t) if t not in special_tokens else [t] for t in data
    ))
    # for special token, not count
    pre_train_freq = Counter(x for x in data if x not in special_tokens)

    # init vocab
    vocab = {i: bytes([i]) for i in range(256)}
    next_vocab_key = 256
    existing_tokens = {bytes([i]) for i in range(256)}
     
    for st in special_tokens: 
        st_bytes = st.encode('utf-8')
        if st_bytes not in existing_tokens and len(vocab) < vocab_size: 
            vocab[next_vocab_key] = st_bytes
            existing_tokens.add(st_bytes)
            next_vocab_key += 1

    # create byte to unicode map and inverse map
    # bytes_to_unicode_map = gpt2_bytes_to_unicode_local()
    # unicode_to_bytes_map = {v: bytes([k]) for k, v in bytes_to_unicode_map.items()}

    byte_freq = {
        tuple(b.encode('utf-8') for b in word): freq
        for word, freq in pre_train_freq.items()
    } 

    epochs = vocab_size - (len(existing_tokens))

    merged_dict = byte_freq

    merge = []

    for _ in range(epochs): 
        merged_dict = bpe_merge(merged_dict, merge)

    
    unique_tokens = {token for tokens in merged_dict.keys() for token in tokens} 

    print(f"the unique_tokens size = {len(unique_tokens)}, the vocab's size = {len(vocab.items())}\
          the sum = {len(unique_tokens) + len(vocab.items())}")
    for token in unique_tokens: 
        if token not in existing_tokens and len(vocab) < vocab_size: 
            vocab[next_vocab_key] = token
            existing_tokens.add(token)
            next_vocab_key += 1
    print(f"the vocab_size = {len(vocab.items())}") 
    
    return vocab, merge 
    


    
# new_data = {
#     tuple([c.encode() for c in "hello"]): 3,
#     tuple([c.encode() for c in "hell"]): 2,
#     tuple([c.encode() for c in "yellow"]): 1,
# }

# merge_ops = []
# print("Before merge:")
# for word, freq in new_data.items():
#     print(word, freq)

# # 调用 merge
# for _ in range(3):
#     new_data = bpe_merge(new_data, merge_ops)

# print("\nAfter merge:")
# for word, freq in new_data.items():
#     print(word, freq)

# print("\nMerge history:", merge_ops)