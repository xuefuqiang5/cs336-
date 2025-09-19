
string = {"hello": 4, "world": 3, "nihao": 4}
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


bytes_to_unicode_map = gpt2_bytes_to_unicode_local()
unicode_to_bytes_map = {v: bytes([k]) for k, v in bytes_to_unicode_map.items()}

byte_freq = {
    tuple(b.encode('utf-8') for b in word): freq
    for word, freq in string.items()
} 
print(byte_freq.items())