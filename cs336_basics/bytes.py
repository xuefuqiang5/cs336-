import regex as re
from itertools import chain
from collections import Counter
data = "hello, world!, nihao woshi xue wenqi <|endoftext|> xue wenqi 0001"
special_tokens = ["<|endoftext|>"]
pat = "(" + "|".join(map(re.escape, special_tokens)) + ")"
data = re.split(pat, data)
pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
data = list(chain.from_iterable(
    re.findall(pat, t) if t not in special_tokens else [t] for t in data
))
print(data)

pre_train_freq = Counter(x for x in data if x not in special_tokens)
print(pre_train_freq.items())


data = {"helo": 1, "world": 2, "“": 3}
data = {word.encode('utf-8'): freq for word, freq in data.items()}
print(data.items())

byte_freq = {tuple(bytes([b]) for b in word): freq for word, freq in data.items()}
print(byte_freq.items())