import regex as re
from collections import Counter
from itertools import chain
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import heapq
from tqdm import tqdm
import json
import os
import time

def update(pair_info, best_pair, word_count):
    affected_words = list(pair_info[best_pair]["word"])
    affected_pairs = set()
    
    replacement = best_pair[0] + best_pair[1]

    for word in affected_words:
        if word not in word_count:
            continue
            
        freq = word_count.pop(word)

        for i in range(len(word) - 1):
            p = (word[i], word[i+1])
            pair_info[p]["freq"] -= freq
            pair_info[p]["word"].discard(word)
            affected_pairs.add(p)

        new_word_list = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                new_word_list.append(replacement)
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1
        new_word = tuple(new_word_list)

        word_count[new_word] = word_count.get(new_word, 0) + freq

        for i in range(len(new_word) - 1):
            p = (new_word[i], new_word[i+1])
            pair_info[p]["freq"] += freq
            pair_info[p]["word"].add(new_word)
            affected_pairs.add(p)

    pair_info[best_pair]["freq"] = 0
    pair_info[best_pair]["word"].clear()
    affected_pairs.add(best_pair)
    return affected_pairs

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
    print("Starting pre tokenizer")
    start = time.perf_counter()
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
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.6f} seconds")
    vocab = {}
    merges = []
    word_count = byte_counter
    pair_info = defaultdict(
        lambda: {
            "freq": 0,
            "word": set()
        }
    )
    for word, freq in byte_counter.items(): 
        for i in range(len(word) - 1): 
            pair = (word[i], word[i+1])
            pair_info[pair]["freq"] += freq
            pair_info[pair]["word"].add(word) 

    class ReverseByteWrapper:
        __slots__ = ['pair']  

        def __init__(self, pair: tuple):
            self.pair = pair

        def __lt__(self, other):
            return self.pair > other.pair

        def __eq__(self, other):
            return self.pair == other.pair

        def __repr__(self):
            return f"ReverseByteWrapper({self.pair})"
    heap = [
        (-info["freq"], ReverseByteWrapper(pair)) 
        for pair, info in pair_info.items() 
        if info["freq"] > 0
    ]
    heapq.heapify(heap)
    for _ in tqdm(range(vocab_size - len(special_tokens) - 256)): 
        best_pair = None
        while True: 
            neg_freq, wrapper = heapq.heappop(heap)
            best_pair = wrapper.pair
            if -neg_freq == pair_info[best_pair]["freq"]:
                break
        merges.append(best_pair)
        affected_pair = update(pair_info, best_pair, word_count)
        for p in affected_pair: 
            heapq.heappush(heap, (-pair_info[p]["freq"], ReverseByteWrapper(p)))
    vocab_list = [bytes([i]) for i in range(256)] + [s.encode('utf-8') for s in special_tokens] + [p[0] + p[1] for p in merges]
    vocab  = {i: v for i, v in enumerate(vocab_list)}
    return vocab, merges

