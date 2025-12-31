import regex as re
from itertools import chain
import json
from cs336_basics.trian_vocab_merges import BYTE_DECODER
from collections.abc import Iterable, Iterator

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
        self.BYTE_DECODER = BYTE_DECODER

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
       # --- 1. 加载并还原 Vocab ---
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        
        # 目标：将 {"Ġt": 123} 还原为 {123: b' t'}
        # 注意：json 加载后的 key 永远是 str，value 是 int (id)
        vocab = {}
        for idx, token_str in vocab_json.items():
            # 使用 BYTE_DECODER 将映射字符串还原为原始字节流
            # 例如: "Ġt" -> [32, 116] -> b' t'
            token_bytes = bytes([BYTE_DECODER[char] for char in token_str])
            vocab[int(idx)] = token_bytes

        # --- 2. 加载并还原 Merges ---
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()
                if len(parts) == 2:
                    # 将 "Ġ t" 还原为 (b' ', b't')
                    p1 = bytes([BYTE_DECODER[char] for char in parts[0]])
                    p2 = bytes([BYTE_DECODER[char] for char in parts[1]])
                    merges.append((p1, p2))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)   

    def merge_word(self, word: bytes) -> list[bytes]: 
        merged_word = list(word)
        merged_word = [bytes([b]) for b in merged_word]
        merges = {pair:i for i, pair in enumerate(self.merges)}
        while True: 
            pairs = [(merged_word[i], merged_word[i+1]) for i in range(0, len(merged_word) - 1)] 
            candidate_merged_pair = None
            idx = -1
            for j, pair in enumerate(pairs): 
                if pair in merges:
                    if candidate_merged_pair == None or merges[pair] < merges[candidate_merged_pair]: 
                        candidate_merged_pair = pair
                        idx = j
            if candidate_merged_pair == None: 
                break 
            merged_word = merged_word[:idx] + [candidate_merged_pair[0] + candidate_merged_pair[1]] + merged_word[idx+2:]
        return merged_word


    def encode(self, text: str) -> list[int]: 
        pretokens = pre_tokenizer(text, self.special_tokens)
        tokens = [w.encode('utf-8') for w in pretokens]
        interger_seq = []
        reverse_vocab = {v:k for k, v in self.vocab.items()}
        if self.special_tokens is not None:
            special_tokens = [s.encode('utf-8') for s in self.special_tokens]
        else: 
            special_tokens = None
        for w in tokens: 
            if special_tokens is not None and w in special_tokens: 
                interger_seq.append(reverse_vocab[w])
                continue
            t = self.merge_word(w)
            for b in t: 
                interger_seq.append(reverse_vocab[b]) 
        return interger_seq
    

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (e.g., file lines).
        This avoids loading large text into memory.
        """
        for chunk in iterable:
            pretokens = pre_tokenizer(chunk, self.special_tokens)
            words = [w.encode("utf-8") for w in pretokens]
            reverse_vocab = {v: k for k, v in self.vocab.items()}
            special_bytes = None
            if self.special_tokens:
                special_bytes = [s.encode("utf-8") for s in self.special_tokens]
            for w in words:
                if special_bytes is not None and w in special_bytes:
                    yield reverse_vocab[w]
                    continue
                merged = self.merge_word(w)
                for b in merged:
                    yield reverse_vocab[b]

TOKENIZER = BPETokenizer.from_files("cs336_basics/Tokenizer/vocab.json", "cs336_basics/Tokenizer/merges.txt")