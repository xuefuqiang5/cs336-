import regex as re
from itertools import chain
import json

class Tokenizer: 
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
    ):
        # vocab: dict[str, str] | dict[str, bytes]
        if isinstance(next(iter(vocab.values())), bytes):
            self.vocab = vocab
        else:
            self.vocab = {k: v.encode('utf-8') for k, v in vocab.items()}

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # merges: list[tuple[str, str]] | list[tuple[bytes, bytes]]
        if isinstance(merges[0][0], bytes):
            self.merges = merges
        else:
            self.merges = [(w1.encode('utf-8'), w2.encode('utf-8')) for (w1, w2) in merges]

        self.special_tokens = special_tokens
    
    @classmethod
    def from_files(
            cls, 
            vocab_filepath: str, 
            merges_filepath: str, 
            special_tokens: list[str] | None = None
    ): 
        with open(vocab_filepath, 'r', encoding='utf-8') as f: 
            vocab = json.load(f)

        with open(merges_filepath, 'r', encoding='utf-8') as f: 
            merge = [tuple(line.strip().split(' ')) for line in f if not line.startswith('#')]
        
        special_tokens = special_tokens if special_tokens is not None else None

        if isinstance(next(iter(vocab.values())), int):
            vocab = {v: k for k, v in vocab.items()}

        return cls(vocab, merge, special_tokens) 

    def encode(self, text: str) -> list[int]: 
        
        if self.special_tokens:
            pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            text = re.split(pat, text)

        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            text = list(chain.from_iterable(
                re.findall(pat, t) if t not in self.special_tokens else [t] for t in text
            ))

        else: 

            text = re.findall(pat, text)

        print(f"the text = {text}")
        bytes_lists = []
        for word in text: 

            if self.special_tokens and word in self.special_tokens:
                bytes_lists.append(word.encode('utf-8'))
                continue

            bytes_list, i = [], 0

            word = [bytes([b]) for b in word.encode('utf-8')]

            if word[0] == b' ':
                n = word[0] + word[1]
                word[0: 2] = [n] 

            print(f'the bytes_list = {word}')
            bytes_list = word
            merges = {pair: i for i, pair in enumerate(self.merges)}
            while True:
                candidate_pair = []

                for i in range(len(bytes_list) - 1):

                    pair = (bytes_list[i],  bytes_list[i+1])
                    if pair in merges.keys(): 
                        candidate_pair.append((pair, i, merges[pair]))

                if not candidate_pair:
                    break

                best_pair = min(candidate_pair, key = lambda x: x[2])
                print(f'the merge pair is {best_pair[0]}')
                
                bytes_list[best_pair[1]: best_pair[1] + 2] = [best_pair[0][0] + best_pair[0][1]]

            bytes_lists.append(bytes_list)
        tokens = list(chain.from_iterable(bytes_lists))
        
        print(f'the tokens = {tokens}')
        try:
            tokens = [self.reverse_vocab[token] for token in tokens]

        except KeyError as e:  
            raise ValueError(f"Token {e.args[0]} not found in reverse_vocab")
    
        return tokens

    def decode(self, ids: list[int]) -> str: 

        bytes_string = b"".join(self.vocab[idx] for idx in ids)
        return bytes_string.decode('utf-8', errors='replace') 

    def print_vocab(self): 
        i = 0
        print('the vocab is')
        for item in self.vocab.items():
            if i == 5:
                break
            i += 1
            print(item)

        print('reverse vocab is') 

        j = 0
        for x in self.reverse_vocab.items(): 
            if j == 5: 
                break
            
            j += 1
            print(x)
    
    def find_idx(self, token): 
        try:
            print(self.reverse_vocab[token])

        except: 
            print(f"{token} not fount in reverse_vocab")


# test_string = "hello, world, !!!!, TypeError: 'NoneType' object is not iterable"
# vocab_filepath = "/Users/xuewenqi/code/cs336/assignment1-basics/tests/fixtures/gpt2_vocab.json"
# merges_filepath = "/Users/xuewenqi/code/cs336/assignment1-basics/tests/fixtures/gpt2_merges.txt" 
# tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath)
# tokenizer.print_vocab()
# print(tokenizer.encode(test_string))
# tokenizer.find_idx(b' ')
