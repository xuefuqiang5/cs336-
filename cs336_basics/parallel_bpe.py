import regex as re
from collections import Counter
from itertools import chain
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import heapq
# def update(pair_info, best_pair, word_count):

#     affected_pairs = set()
#     affected_words = list(pair_info[best_pair]["word"])

#     for word in affected_words:
#         freq = word_count[word]

#         new_word = []
#         i = 0

#         removed_pairs = Counter()
#         added_pairs = Counter()

#         while i < len(word):
#             if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
#                 if i > 0:
#                     removed_pairs[(word[i-1], word[i])] += 1
#                 if i + 2 < len(word):
#                     removed_pairs[(word[i+1], word[i+2])] += 1

#                 new_word.append(word[i] + word[i+1])
#                 i += 2
#             else:
#                 new_word.append(word[i])
#                 i += 1

#         r = best_pair[0] + best_pair[1]
#         for i in range(len(new_word) - 1):
#             if new_word[i] == r or new_word[i+1] == r:
#                 added_pairs[(new_word[i], new_word[i+1])] += 1

 
#         new_word = tuple(new_word)
#         old_freq = word_count.pop(word)
#         word_count[new_word] = word_count.get(new_word, 0) + old_freq
        
#         for p, cnt in removed_pairs.items():
#             pair_info[p]["freq"] -= cnt*freq
#             pair_info[p]["word"].discard(word)
#             affected_pairs.add(p)

#         for p, cnt in added_pairs.items():
#             pair_info[p]["freq"] += cnt*freq
#             pair_info[p]["word"].add(new_word)
#             affected_pairs.add(p)

#     pair_info[best_pair]["freq"] = 0
#     pair_info[best_pair]["word"].clear()
#     return affected_pairs

def update(pair_info, best_pair, word_count):
    """
    使用 '先全减、后全增' 策略更新频率，彻底解决重叠 Pair 计数问题。
    """
    # 1. 找到所有包含该 best_pair 的词（当前的词元序列）
    # 使用 list() 拷贝是因为我们在循环中会修改 word_count
    affected_words = list(pair_info[best_pair]["word"])
    
    # 预拼接好合并后的新词元
    replacement = best_pair[0] + best_pair[1]

    for word in affected_words:
        if word not in word_count:
            continue
            
        # 获取该词出现的频率
        freq = word_count.pop(word)

        # --- 第一步：从 pair_info 中完全抹除该词旧序列的贡献 ---
        for i in range(len(word) - 1):
            p = (word[i], word[i+1])
            pair_info[p]["freq"] -= freq
            # 这一步是为了维护索引的干净，虽然耗时但在 Python 集合中是 O(1)
            pair_info[p]["word"].discard(word)

        # --- 第二步：执行 BPE 合并逻辑，生成新序列 ---
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

        # --- 第三步：更新 word_count (重要：处理合并后的碰撞) ---
        # 如果 ('a','b','c') 和 ('ab','c') 在合并 (a,b) 后都变成了 ('ab','c')
        # 它们的频率必须累加，否则会丢失数据
        word_count[new_word] = word_count.get(new_word, 0) + freq

        # --- 第四步：将新序列产生的 Pair 贡献增加到 pair_info ---
        for i in range(len(new_word) - 1):
            p = (new_word[i], new_word[i+1])
            pair_info[p]["freq"] += freq
            pair_info[p]["word"].add(new_word)

    # 显式清理，确保 best_pair 彻底归零（可选，逻辑正确时它自然会归零）
    pair_info[best_pair]["freq"] = 0
    pair_info[best_pair]["word"].clear()
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


    for _ in range(vocab_size - len(special_tokens) - 256): 
        # max_freq = max([info["freq"] for info in pair_info.values()])
        # candidates = [
        #     pair
        #     for pair, info in pair_info.items()
        #     if info["freq"] == max_freq
        # ]
        # best_pair = max(candidates)
        best_pair = max(
            pair_info.items(),
            key=lambda x: (x[1]["freq"], x[0]) # 比较频率，其次比较 Pair 的字典序
        )[0] 
        merges.append(best_pair)
        update(pair_info, best_pair, word_count)
    vocab_list = [bytes([i]) for i in range(256)] + [s.encode('utf-8') for s in special_tokens] + [p[0] + p[1] for p in merges]
    vocab  = {i: v for i, v in enumerate(vocab_list)}
    return vocab, merges