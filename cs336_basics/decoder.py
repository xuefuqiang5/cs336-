import torch
import torch.nn.functional as F
from einops import rearrange
from Tokenizer import BPETokenizer 
from cs336_basics.Transformer import TransformerLm

def get_next_token(logits, temperature, p):
    """
    基于 Top-P (Nucleus) Sampling 获取下一个 token
    :param logits: [batch_size, seq_length, vocab_size]
    :param temperature: 温度系数
    :param p: Top-P 累积概率阈值 (0.0 - 1.0)
    """
    # 通常生成任务只需要最后一个时间步的 logits
    # shape: [batch_size, vocab_size]
    logits = logits[:, -1, :] 
    
    # 处理 temperature 维度 (假设输入是 tensor)
    if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
        temperature = rearrange(temperature, "v -> 1 v")
    
    # 1. 计算概率分布
    probs = F.softmax(logits / temperature, dim=-1)
    
    # 2. 排序 (Sort)
    # torch.sort 返回 (values, indices)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 3. 计算累积概率 (Cumsum)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 4. 生成掩码 (Masking)
    # 移除累积概率超过阈值 p 的 token
    sorted_indices_to_remove = cumulative_probs > p
    
    # 技巧：将掩码向右移动一位，以保留第一个超过阈值的 token
    # (确保至少有一个 token 被选中，防止 p 很小时所有都被 mask 掉)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 5. 映射回原始索引 (Scatter)
    # 将排序后的掩码映射回原始词表顺序
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    
    # 6. 应用掩码并重新归一化
    # 将被移除的 token 概率置为 0
    probs = probs.masked_fill(indices_to_remove, 0.0)
    
    # 重新归一化概率分布 (因为去掉了一部分尾部概率)
    # 加上 1e-8 防止除以零
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    # 7. 采样 (Sampling)
    # shape: [batch_size, 1]
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token 

def generate_text(model, prompt, temperature, p, max_length=100): 
    from Tokenizer import TOKENIZER
    tokenizer = TOKENIZER
    
    device = next(model.parameters()).device
    
    ids_list = tokenizer.encode(prompt)
    ids = torch.tensor([ids_list], dtype=torch.long).to(device) # shape: [1, seq_len]
    
    buffer = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length): # 使用 for 循环防止无限死循环
            logits = model(ids)
            
            next_token = get_next_token(logits, temperature, p)
            
            ids = torch.cat([ids, next_token], dim=-1)
            
            token_id = next_token.item()
            eos_id = tokenizer.encode("<|endoftext|>") 

            if token_id == eos_id[0]: 
                break
            word = tokenizer.decode([token_id])
            
            if "<|endoftext|>" in word:
                break
            
            
            print(word, end="", flush=True) # flush 确保实时打印
            buffer.append(word)
    
    return "".join(buffer) # 通常返回拼接后的完整字符串比返回 list 更好

if __name__ == "__main__": 
    # 1. 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. 初始化模型 (参数必须与训练时完全一致)
    model = TransformerLm(
        vocab_size=10000, 
        context_length=1024,
        num_layers=12, 
        d_model=768, 
        num_heads=12, 
        d_ff=3072, 
        theta=10000.0
    )
    model.to(device)

    # 3. 加载 Checkpoint
    checkpoint_path = "checkpoint_0.pt"
    import os 
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到文件 {checkpoint_path}")
        exit()

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        # map_location 确保在无 GPU 环境也能加载
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # --- 核心修复逻辑 ---
        # 你的报错显示 checkpoint 是包含 "model_state" 的字典
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 处理 'module.' 前缀 (DDP 训练遗留)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
            
        # 加载参数
        msg = model.load_state_dict(new_state_dict, strict=True)
        print(f"Model loaded successfully! {msg}")
        
    except Exception as e:
        print(f"\n加载模型失败: {e}")
        print("建议: 检查 TransformerLm 的定义是否和训练代码中的定义完全一致。")
        exit()

    # 4. 执行生成
    # 你可以在这里修改提示词
    start_prompt = "I can still remember"
    
    # 调整生成参数
    temperature = 0.8  # (0.0 - 1.0+) 控制随机性
    top_p = 0.9        # (0.0 - 1.0) Nucleus sampling
    
    print(start_prompt, end=" ")
    generate_text(
        model=model, 
        prompt=start_prompt, 
        temperature=temperature, 
        p=top_p, 
        max_length=1000
    )