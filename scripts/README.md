# Scripts 使用说明

## 文件结构

```
scripts/
├── ablation/
│   ├── __init__.py
│   └── transformer_variants.py   # 消融实验 Transformer 变体（不修改 basics/）
├── configs/
│   ├── ablation.yaml              # 消融实验 Hydra 配置
│   ├── pretrain.yaml              # 预训练配置
│   ├── tokenizer.yaml             # Tokenizer 配置
│   └── evaluate_cs336_lm.yaml     # 模型评估配置
├── run_ablations.py               # 消融实验运行脚本 ★
├── pretrain_model.py              # 预训练脚本
├── train_tokenizer.py             # Tokenizer 训练脚本
├── eva_pretrain_model.py          # 模型评估脚本
├── mytokenize.py                  # Tokenize 数据脚本
└── README.md                      # 本文件
```

---

## 消融实验（`run_ablations.py`）

运行三类消融实验，使用 wandb 记录并对比学习曲线。

### 快速开始

```bash
# 运行全部 3 组消融实验（共 7 个 run）
python scripts/run_ablations.py

# 只运行单组
python scripts/run_ablations.py ablation_group=layer_norm
python scripts/run_ablations.py ablation_group=pos_emb
python scripts/run_ablations.py ablation_group=ffn
```

### 实验设计

| 消融组 | wandb group | Run | 说明 |
|--------|-------------|-----|------|
| **Ablation 1** | `layer_norm` | `pre-norm-rmsnorm` | 基线：Pre-norm + RMSNorm |
| | | `no-rmsnorm` | 无 RMSNorm（先用 LR=5e-4，发散则自动回退 LR=1e-4）|
| | | `post-norm-rmsnorm` | Post-norm + RMSNorm |
| **Ablation 2** | `pos_emb` | `rope` | 基线：RoPE 位置编码 |
| | | `nope` | 无位置编码（NoPE）|
| **Ablation 3** | `ffn` | `swiglu` | 基线：SwiGLU（d_ff ≈ 8/3 × d_model）|
| | | `silu` | SiLU（d_ff = 4 × d_model，参数量匹配）|

### 模型变体说明

所有变体通过 `AblationTransformerLM` 类动态构建，支持 4 个控制标志：

| 参数 | 值 | 作用 |
|------|-----|------|
| `use_rmsnorm` | `true` / `false` | 是否使用 RMSNorm（false 时替换为 Identity）|
| `norm_style` | `"pre"` / `"post"` | Pre-norm 或 Post-norm |
| `pos_emb` | `"rope"` / `"nope"` | 位置编码类型 |
| `ffn_type` | `"swiglu"` / `"silu"` | FFN 激活函数 |

### wandb 配置

编辑 `scripts/configs/ablation.yaml` 设置 wandb：

```yaml
wandb:
  project: cs336-ablations       # wandb 项目名
  entity: null                   # 设为你的 wandb entity（或 null 使用默认）
```

### 训练超参调整

所有训练参数在 `scripts/configs/ablation.yaml` 的 `training` 段中：

```yaml
training:
  lr: 0.0005          # 学习率
  min_lr: 0.0001      # 最小学习率
  batch_size: 32      # 批次大小
  train_steps: 5000   # 训练步数
  warmup_iters: 500   # 预热步数
  cosine_iters: 5000  # 余弦衰减步数
  context_length: 256 # 上下文长度
  # ... 更多参数见配置文件
```

### No-RMSNorm 自动 LR 回退

`no-rmsnorm` run 先尝试默认 LR=5e-4 训练。若 loss 出现 NaN/Inf，自动：
1. 记录失败日志
2. 以 LR=1e-4 重新初始化模型和 wandb run
3. 重新训练

若两次均发散则跳过并记录错误。

---

## 预训练（`pretrain_model.py`）

标准 Transformer 语言模型预训练，使用 mlflow 记录。

```bash
python scripts/pretrain_model.py
```

配置：`scripts/configs/pretrain.yaml`

---

## Tokenizer 训练（`train_tokenizer.py`）

BPE Tokenizer 训练脚本。

```bash
python scripts/train_tokenizer.py
```

配置：`scripts/configs/tokenizer.yaml`

---

## 模型评估（`eva_pretrain_model.py`）

加载 checkpoint 进行文本生成评估。

```bash
python scripts/eva_pretrain_model.py
```

配置：`scripts/configs/evaluate_cs336_lm.yaml`
