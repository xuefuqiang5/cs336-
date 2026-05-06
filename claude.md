You are an expert PyTorch and ML systems engineer. I am working on CS336 Assignment 1, specifically the ablation experiments for a custom Transformer language model. 

I already have the core Transformer modules implemented in my project directory. I need you to write an elegant Python training orchestrator/script that runs these specific ablation experiments and uses `wandb` to log and compare the results.

### 🎯 Key Requirements:
1. **Non-Intrusive Wandb Integration:** I want to keep my core `train(model, device, args)` function as clean and untouched as possible. Initialize `wandb.init()` before calling the train loop, and ensure wandb logs metrics (like loss, lr) either inside the train loop or via a callback/wrapper. Use `wandb` tags and groups (e.g., `group: "layer_norm_ablation"`) to easily compare runs in the dashboard.
2. **Dynamic Model Instantiation:** Write a script or a setup function that iterates through the ablation configurations and initializes the model dynamically based on `args` or a config dictionary. Do not hardcode the model changes into the training loop.
3. **The 3 Ablation Scenarios to Implement (as sweeps/loops):**
   * **Ablation 1: LayerNorm:** - Run 1: Base Model (Pre-norm RMSNorm).
     - Run 2: No RMSNorm (try default LR, then try a smaller LR if it diverges).
     - Run 3: Post-norm architecture.
   * **Ablation 2: Position Embeddings:**
     - Run 1: Base Model (with RoPE).
     - Run 2: NoPE (No Position Embeddings).
   * **Ablation 3: FFN Activations:**
     - Run 1: Base Model (SwiGLU, d_ff = 8/3 * d_model).
     - Run 2: SiLU (No GLU, d_ff = 4 * d_model to match parameter counts).

### 🛠️ What to generate:
Please provide:
1. The **Python experiment running script** (e.g., `run_ablations.py`) that sets up these configurations, handles `wandb` initialization with proper grouping, instantiates the correct model variants (assuming the classes `Transformer`, `TransformerPostNorm`, `NoPETransformer`, etc., or configurable flags in a single `Transformer` class exist), and calls my existing `train()` function.
2. If necessary, a brief demonstration of how I should structure my `Transformer` class `__init__` to accept these ablation flags (e.g., `use_rmsnorm=True`, `norm_style='pre'`, `pos_emb='rope'`, `ffn_type='swiglu'`).

Ensure the code is robust, well-commented, and explicitly logs to wandb so I can generate learning curves comparing these specific configurations.
