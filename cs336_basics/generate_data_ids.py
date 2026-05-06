import os
import numpy as np
import time
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- 1. 子进程环境配置 ---
_tokenizer = None

def _init_worker(vocab_path, merges_path, special_tokens):
    """
    每个子进程启动时，独立加载词表到自己的内存中。
    这避免了主进程重复打包发送大对象。
    """
    global _tokenizer
    from cs336_basics.Tokenizer import BPETokenizer
    _tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

def _encode_worker_fast(text_chunk):
    """
    加速版 Worker：直接在子进程里转成 numpy 字节流返回。
    """
    if not text_chunk:
        return None
    ids = _tokenizer.encode(text_chunk)
    if not ids:
        return None
    # 在子进程里直接转成 uint32，主进程收到的就是原始字节块
    return np.array(ids, dtype=np.uint32)



def generate_idx_data_v3(
    input_filepath: str,
    output_dir: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list = ["<|endoftext|>"],
    num_workers: int = os.cpu_count()
):
    os.makedirs(output_dir, exist_ok=True)
    bin_file = os.path.join(output_dir, "data.bin")

    print(f"[*] 正在读取并进行大块打包...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        # 依然按故事切分
        raw_stories = f.read().split("<|endoftext|>")
    
    # --- 关键改进：将微小任务合并为大任务 ---
    group_size = 5000  # 每 5000 个小故事合成一个大块进行一次通信
    grouped_chunks = [
        "<|endoftext|>".join(raw_stories[i : i + group_size]) 
        for i in range(0, len(raw_stories), group_size)
    ]
    
    print(f"[*] 任务重组完成：271万个小任务 -> {len(grouped_chunks)} 个大任务")
    # ---------------------------------------

    token_count = 0
    start_time = time.time()

    with open(bin_file, "wb") as f_out:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(vocab_path, merges_path, special_tokens)
        ) as executor:
            
            # 由于任务已经打包，chunksize 设为 1 即可
            results = executor.map(_encode_worker_fast, grouped_chunks, chunksize=1)
            
            for data_array in tqdm(results, total=len(grouped_chunks), desc="Grouped Encoding"):
                if data_array is not None:
                    f_out.write(data_array.tobytes())
                    token_count += len(data_array)

    duration = time.time() - start_time
    print(f"\n[√] 处理完成！耗时: {duration:.2f}s")
    # 预估此版本耗时在 3-8 分钟之间·
import os
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_chunks_fast(file_path, chunk_size_mb=50):
    """
    高效读取器：按块读取，并确保不截断 <|endoftext|> 分隔符
    """
    delimiter = "<|endoftext|>"
    chunk_size = chunk_size_mb * 1024 * 1024
    with open(file_path, 'r', encoding='utf-8') as f:
        remainder = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if remainder:
                    yield remainder
                break
            
            # 加上上一次剩下的部分
            combined = remainder + chunk
            # 找到最后一个分隔符的位置
            last_delim_idx = combined.rfind(delimiter)
            
            if last_delim_idx != -1:
                # 这一块是完整的故事集合
                yield combined[:last_delim_idx + len(delimiter)]
                # 剩下不完整的部分留给下一次
                remainder = combined[last_delim_idx + len(delimiter):]
            else:
                # 如果这一块里连一个分隔符都没有，全部存入 remainder
                remainder = combined

def generate_idx_data_v4(
    input_filepath: str,
    output_dir: str,
    vocab_path: str,
    merges_path: str,
    num_workers: int = os.cpu_count()
):
    os.makedirs(output_dir, exist_ok=True)
    bin_file = os.path.join(output_dir, "data.bin")

    # 1. 预估任务数（用于 tqdm 显示，不耗时）
    file_size = os.path.getsize(input_filepath)
    chunk_size_mb = 50 
    total_expected = file_size // (chunk_size_mb * 1024 * 1024)

    print(f"[*] 启动流式处理：每块 {chunk_size_mb}MB，共计约 {total_expected} 块")

    token_count = 0
    start_time = time.time()

    with open(bin_file, "wb") as f_out:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker, # 保持之前的 _init_worker 不变
            initargs=(vocab_path, merges_path, ["<|endoftext|>"])
        ) as executor:
            
            # 使用迭代器而不是列表，主进程瞬间启动
            # 这里的 chunks 是生成器，不会产生 271 万个对象
            chunks = get_chunks_fast(input_filepath, chunk_size_mb=chunk_size_mb)
            
            results = executor.map(_encode_worker_fast, chunks, chunksize=1)
            
            # 此时 tqdm 会立刻出现并开始计数
            for data_array in tqdm(results, total=total_expected, desc="Streaming Encoding"):
                if data_array is not None:
                    f_out.write(data_array.tobytes())
                    token_count += len(data_array)

    print(f"\n[√] 处理完成！耗时: {time.time() - start_time:.2f}s")
if __name__ == "__main__":
    # 使用你自己的路径
    generate_idx_data_v4(
        input_filepath="data/TinyStoriesV2-GPT4-train.txt",
        output_dir="data",
        vocab_path="cs336_basics/Tokenizer/vocab.json",
        merges_path="cs336_basics/Tokenizer/merges.txt",
        num_workers=os.cpu_count() # 使用全部 36 核
    )