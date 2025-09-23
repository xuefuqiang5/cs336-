import regex as re
from itertools import chain

class Tokenizer: 
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
    ):
        self.vocab = vocab 
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(
            cls, 
            vocab_filepath: str, 
            merges_filepath: str, 
            special_tokens: list[str] | None = None
    ): 
        pass
    
    def encode(self, text: str) -> list[int]: 
        pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
        text = re.split(pat, text)

        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text = list(chain.from_iterable(
            re.findall(pat, t) if t not in self.special_tokens else [t] for t in text
        ))
        bytes_lists = []
        for word in text: 

            new_word = [] 
            bytes_list = [bytes([b]) for b in word.encode('utf-8')]

            for i in range(len(bytes_list) - 1): 

                new_token = bytes_list[i] + bytes_list[i+1]

                if (bytes_list[i], bytes_list[i+1]) in self.merges: 

                    del bytes_list[i+1]
                    del bytes_list[i]
                    bytes_list.insert(i, new_token)
                    i = 0

            bytes_lists.append(bytes_list)

        tokens = list(chain.from_iterable(bytes_lists))

        tokens = [self.reverse_vocab[token] for token in tokens]


    def decode(self, ids: list[int]) -> str: 
        pass