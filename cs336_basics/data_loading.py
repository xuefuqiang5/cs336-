import torch
import numpy as np
from cs336_basics.Tokenizer import BPETokenizer, split2chunks

def data_loading(data, batch_size, context_length, device): 
    n_tokens = len(data)
    if n_tokens <= context_length:
        return None
    max_start = len(data) - context_length 
    starts = np.random.randint(0, max_start, size=batch_size)
    x, y = [], []
    for i in starts: 
        x.append(data[i: context_length+i])
        y.append(data[i+1: context_length+i+1])
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y
    

def generate_idx_data(input_filepath: str, output_dir: str, tokenizer: BPETokenizer):  
    os.makedirs(output_dir, exist_ok=True) 
    bin_file = os.path.join(output_dir, "data.bin")
    token_count = 0
    with open(bin_file, "wb") as f: 
        chunks = split2chunks(input_filepath, "<|endoftext|>")
        for chunk in chunks: 
            ids = tokenizer.encode(chunk)
            if ids > 0:
                data = np.array(ids, dtype=np.int32)
                f.write(ids)
                token_count += len(data)
    
    print(f"处理完成！总计 Token 数量: {token_count}")
    print(f"文件大小: {os.path.getsize(bin_file) / (1024**2):.2f} MB")

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

# 全局变量或包装函数，方便多进程调用
def _encode_worker(tokenizer, chunk):
    """单进程工作函数：编码一个文本块"""
    # 假设 tokenizer.encode 接受字符串并返回 list[int]
    ids = tokenizer.encode(chunk)
    # 如果 split2chunks 移除了 <|endoftext|>，这里可以手动加上
    # ids.append(tokenizer.special_tokens["<|endoftext|>"])
    return ids

def generate_idx_data_parallel(
    input_filepath: str, 
    output_dir: str, 
    tokenizer: BPETokenizer, 
    num_workers: int = os.cpu_count(),
    buffer_size: int = 1024 * 1024 * 5 # 每个进程处理 5MB 的块
):
    os.makedirs(output_dir, exist_ok=True)
    bin_file = os.path.join(output_dir, "data.bin")
    token_count = 0

    # 1. 获取 chunks 生成器
    # 注意：确保你的 split2chunks 返回的是字符串列表/生成器
    from cs336_basics.Tokenizer import split2chunks
    chunks = split2chunks(input_filepath, "<|endoftext|>", buffer_size=buffer_size)

    # 2. 使用进程池
    # 建议使用 imap 或 map 以保持数据的原始顺序
    print(f"使用 {num_workers} 个核心进行并发编码...")
    
    with open(bin_file, "wb") as f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 使用 partial 锁定 tokenizer 参数
            worker_func = partial(_encode_worker, tokenizer)
            
            # 使用 map 保证返回的 ids 顺序与 chunks 顺序一致
            # chunk_size=1 让进程池从生成器中一个一个取 chunk 避免内存溢出
            results = executor.map(worker_func, chunks, chunksize=1)
            
            for ids in tqdm(results, desc="Encoding and Writing"):
                if len(ids) > 0:
                    # 转换并写入二进制
                    data = np.array(ids, dtype=np.uint32) # 使用 uint32 更省空间且足够
                    f.write(data.tobytes())
                    token_count += len(data)

    print(f"\n[完成] 总计 Token: {token_count}")
    print(f"二进制文件位置: {bin_file}")
    print(f"文件大小: {os.path.getsize(bin_file) / (1024**2):.2f} MB")


if __name__ == "__main__": 
    input_filepath = "/Users/xuewenqi/code/cs336/cs336-/data/data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "/Users/xuewenqi/code/cs336/cs336-/data"
    tokenizer = BPETokenizer.from_files("cs336_basics/Tokenizer/vocab.json", "cs336_basics/Tokenizer/merges.txt", special_tokens="<|endoftext|>")
    generate_idx_data_parallel(input_filepath, output_dir, tokenizer)