import numpy as np
import torch
import os
class DataLoader:
    def __init__(self, data_path, batch_size, context_length, device, dtype=np.uint32):
        """
        参数:
        - data_path: data.bin 的路径
        - batch_size: 每个 batch 的大小
        - context_length: 序列长度 (L)
        - device: 'cuda' 或 'cpu'
        - dtype: 必须与你生成 bin 文件时一致 (通常是 np.uint32)
        """
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        
        # 1. 使用 mmap 挂载文件（只读模式）
        # self.data 现在的行为就像一个超大的 list/ndarray，但数据在磁盘上
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
            
        self.data = np.memmap(data_path, dtype=dtype, mode='r')
        self.total_tokens = len(self.data)
        
        print(f"成功加载数据集，总计 Token 数: {self.total_tokens:,}")

    def __iter__(self):
        """
        使 DataLoader 可迭代，通常在训练循环中使用
        """
        while True:
            # 调用你已有的 data_loading 函数
            # 注意：np.memmap 对象可以直接作为参数传给 data_loading，
            # 只要你的 data_loading 内部是用索引取值的（如 data[i:i+L]）
            yield self.get_batch()
    def get_len(self):
        return len(self.data)
    def get_batch(self):
        """
        封装调用 data_loading 的逻辑
        """
        from cs336_basics.data_loading import data_loading
        
        # 直接将 memmap 数组传进去
        # 你的 data_loading 应该支持从一个类似数组的对象中采样
        x, y = data_loading(
            self.data, 
            self.batch_size, 
            self.context_length, 
            self.device
        )
        return x, y