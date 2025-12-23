import regex as re
from collections import Counter
from itertools import chain
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count
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

def pre_tokenizer(text: str, special_tokens: list[str]):
    if special_tokens:
        # longest match first
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        # (A|BB|CCC)
        pat = "(" + "|".join(map(re.escape, special_tokens)) + ")"
        parts = re.split(pat, text)
    else:
        parts = [text]

    # Now apply normal GPT-2 regex tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    out = []
    for p in parts:
        if not p: 
            continue
        
        # if p is exactly a special token → keep it as-is
        if special_tokens and p in special_tokens:
            out.append(p)
            continue
        
        out.extend(re.findall(PAT, p))
    return out
def pre_tokenizer_worker(args):
    chunk, special_tokens = args
    special_set = set(special_tokens)

    tokens = pre_tokenizer(chunk, special_tokens)
    return Counter(
        tuple(bytes([b]) for b in tok.encode('utf-8')) for tok in tokens
        if tok not in special_set
    )
def train_bep(input_path, vocab_size, special_tokens): 
    chunks = split2chunks(input_path, special_tokens[0])

    n_proc = cpu_count()

    byte_counter = Counter()

    with Pool(processes=n_proc) as pool:
        for local_counter in pool.imap_unordered(
            pre_tokenizer_worker,
            ((chunk, special_tokens) for chunk in chunks),
            chunksize=1
        ):
            byte_counter.update(local_counter)
    
    # count.type = dict[tuple[bytes, bytes], int]
    count = Counter()
    for word, freq in byte_counter.items():
        for i in range(len(word) - 1):
            count[(word[i], word[i+1])] += freq
    
    merged_dict = byte_counter 
    vocab = {i: bytes([i]) for i in range(256)}
    next_idx = 256

    for i, t in enumerate(special_tokens):
        vocab[i+next_idx] = t.encode('utf-8')

    next_idx = len(vocab)
    merges = []
    for _ in range(vocab_size - len(vocab)): 
        max_freq = max([freq for freq in count.values()])
        candidates = [pair for pair, freq in count.items() if freq == max_freq]
        best_pair = max(candidates)
        merges.append(best_pair)
        count[best_pair] = 0
        merged_dict = bpe_merge(merged_dict, best_pair, count)

    assert len(merges) == (vocab_size - len(vocab))

    for i, pair in enumerate(merges): 
        vocab[i+next_idx] = pair[0] + pair[1]

    return vocab, merges 

