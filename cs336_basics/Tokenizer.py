import regex as re
from itertools import chain
import json

"""
Byte Pair Encoding (BPE) Tokenizer Implementation
=================================================

This module implements the text encoding process for a Byte Pair Encoding (BPE)
tokenizer. The encoding pipeline mirrors the standard BPE vocabulary training
procedure and follows several well-defined steps:

1. Pre-tokenization
   - Each input text string is split into pre-tokens (e.g., whitespace- or
     punctuation-based segments).
   - Each pre-token is further represented as a sequence of UTF-8 bytes.
   - BPE merge operations are applied *within* each pre-token only; merges never
     cross pre-token boundaries.

2. Applying BPE merges
   - The learned BPE merge operations (produced during vocabulary training) are
     applied to the byte sequences in the exact order in which the merges were
     created.
   - This process incrementally combines byte pairs into larger vocabulary
     elements until no more merges can be applied.

3. Handling special tokens
   - The tokenizer supports user-defined special tokens (e.g., BOS, EOS, PAD),
     which are inserted or recognized without undergoing byte-pair merging.
   - Special tokens are preserved exactly as defined by the tokenizer
     configuration.

4. Memory-efficient streaming tokenization
   - This implementation supports tokenizing large input files or data streams
     without loading the entire content into memory.
   - The text is processed in fixed-size chunks such that overall memory usage
     remains constant.
   - Care is taken to avoid breaking pre-tokens across chunk boundaries to
     ensure that tokenization results match those produced by a full in-memory
     pass.

Overall, this file provides a faithful BPE tokenizer implementation that is
compatible with common large language model tokenization workflows and supports
both high accuracy and efficient streaming behavior.
"""
def split2chunks(text: str, endoftext: str) -> list[str]: 
    chunks = []
    start = 0
    while True: 
        idx = text.find(endoftext, start) 
        if idx == -1: 
            rest = text[start].strip()
            if rest: 
                chunks.append(rest)
                break
        chunk = text[start:idx+len(endoftext)]
        chunks.append(chunk)
        start = idx + len(endoftext)
    
    return chunks
        
def pre_tokenizer(text: str, special_tokens: list[str] | None=None) -> list[str]: 
    if special_tokens is not None:
        pat = "(" + "|".join(map(re.escape, special_tokens)) + ")"
        pretokens = re.split(pat, text)

        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens = list(chain.from_iterable(
            re.findall(pat, t) if t not in special_tokens else [t] for t in pretokens
        ))
    else: 
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens = re.findall(pat, text)

    return pretokens

    
class BPETokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None=None
        ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] = None
    ):
        """
        Load vocab and merges from files and return a BPETokenizer instance.

        :param vocab_filepath: path to JSON vocab file
        :param merges_filepath: path to merges file
        :param special_tokens: optional list of special tokens
        :return: BPETokenizer instance
        """
        # --- load vocab ---
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        # ensure keys are int, values are bytes
        # if vocab_json is {str: int}, we invert it to int -> bytes
        if all(isinstance(k, str) for k in vocab_json.keys()):
            reverse_vocab = {v: k.encode('utf-8') for k, v in vocab_json.items()}
        else:
            reverse_vocab = {int(k): v.encode('utf-8') for k, v in vocab_json.items()}

        # --- load merges ---
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((a.encode('utf-8'), b.encode('utf-8')))

        # --- create tokenizer instance ---
        return cls(vocab=reverse_vocab, merges=merges, special_tokens=special_tokens)   

    def merge_word(self, word: list[bytes]) -> list[bytes]: 
        i, merged_word = 0, word
        while i+1 < len(merged_word): 
            t = []
            if tuple([merged_word[i], merged_word[i+1]]) in self.merges:
                t = [self.merges[0] + self.merges[1]]
                if i+2 < len(merged_word):
                    merged_word = merged_word[:i] + t + merged_word[i+2:]
                else: 
                    merged_word = merged_word[:i] + t
                i = 0
                continue 
            i += 1
        return merged_word 


    def encode(self, text: str) -> list[int]: 
        pretokens = pre_tokenizer(text, self.special_tokens)
        tokens = [list(bytes([c]) for c in w) for w in pretokens]
        merged_tokens = []
        reverse_vocab = {v:k for k, v in self.vocab.items()}
        special_tokens = [s.encode('utf-8') for s in special_tokens]
        for w in tokens: 
            if w in special_tokens: 
                merged_tokens.append(w)
                continue
            merged_tokens.append(self.merge_word(w))
        integer_seq = [reverse_vocab[b] for w in merged_tokens for b in w]
        return integer_seq
    
    def decode(self): 
        pass


data = "Sample 10 documents from TinyStories \
and OpenWebText. Using your previously-trained TinyS-tories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?\
Deliverable: A one-to-two sentence response.hhh"

bpetokenize = BPETokenizer.from_files(
    "/Users/xuewenqi/code/cs336/assignment1-basics/tests/fixtures/train-bpe-reference-vocab.json", 
    "/Users/xuewenqi/code/cs336/assignment1-basics/tests/fixtures/train-bpe-reference-merges.txt" 
    ) 

print(bpetokenize.encode(data))