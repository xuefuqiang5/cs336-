import regex as re
from itertools import chain
from collections import Counter
data = "hello, world!, nihao woshi xue wenqi <|endoftext|> xue wenqi 0001"
parts = data.split("<|endoftext|>")
print(parts, type(parts))

print(type(Counter()))