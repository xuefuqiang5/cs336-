"""
Ablation variants of Transformer modules.

This module builds on top of basics.transformer and basics.nn_utils
to provide configurable Transformer variants for ablation experiments:

  - norm_style: 'pre' (pre-norm RMSNorm) or 'post' (post-norm)
  - use_rmsnorm: True (RMSNorm) or False (nn.Identity)
  - pos_emb: 'rope' (RotaryEmbedding) or 'nope' (no positional encoding)
  - ffn_type: 'swiglu' (SwiGLU, d_ff=8/3*d_model) or 'silu' (SiLU, d_ff=4*d_model)
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

# Re-use the building blocks from basics — no modifications needed to basics/
from basics.transformer import (
    Embedding,
    RotaryEmbedding,
    CausalMultiHeadSelfAttention,
    scaled_dot_product_attention,
    softmax,
)
from basics.nn_utils import Linear, RMSNorm, SwiGLU, SiLU

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation-aware TransformerBlock
# ---------------------------------------------------------------------------

class AblationTransformerBlock(nn.Module):
    """A single Transformer layer with ablation knobs.

    Args:
        d_model: Model dimensionality.
        num_heads: Number of attention heads (d_model must be divisible by num_heads).
        d_ff: Feed-forward inner dimensionality.
        positional_encoder: RoPE module, or None for NoPE.
        use_rmsnorm: If False, replaces RMSNorm with nn.Identity.
        norm_style: 'pre' for pre-norm, 'post' for post-norm.
        ffn_type: 'swiglu' or 'silu'.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding | None,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.norm_style = norm_style

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        elif ffn_type == "silu":
            # SiLU internally sets d_ff = 4 * d_model to match param counts
            self.ffn = SiLU(d_model=d_model)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_style == "pre":
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        elif self.norm_style == "post":
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        else:
            raise ValueError(f"Unknown norm_style: {self.norm_style}")
        return x


# ---------------------------------------------------------------------------
# Ablation-aware Transformer LM
# ---------------------------------------------------------------------------

class AblationTransformerLM(nn.Module):
    """A Transformer language model with ablation knobs.

    Args:
        vocab_size: Vocabulary size.
        context_length: Maximum sequence length.
        d_model: Model dimensionality.
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads.
        d_ff: Feed-forward inner dimensionality.
        rope_theta: Theta parameter for RoPE (unused when pos_emb='nope').
        use_rmsnorm: Whether to use RMSNorm in each block.
        norm_style: 'pre' or 'post' norm placement.
        pos_emb: 'rope' or 'nope' (no positional embeddings).
        ffn_type: 'swiglu' or 'silu'.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        pos_emb: str = "rope",
        ffn_type: str = "swiglu",
    ):
        self.config = {
            k: v
            for k, v in locals().items()
            if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = Embedding(vocab_size, d_model)

        if pos_emb == "rope":
            d_head = d_model // num_heads
            self.positional_encoder = RotaryEmbedding(
                context_length=context_length, dim=d_head, theta=rope_theta
            )
        elif pos_emb == "nope":
            self.positional_encoder = None
        else:
            raise ValueError(f"Unknown pos_emb: {pos_emb}")

        self.layers = nn.ModuleList(
            [
                AblationTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                    use_rmsnorm=use_rmsnorm,
                    norm_style=norm_style,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer-norm (used only in pre-norm; post-norm already has per-block norm)
        if use_rmsnorm:
            self.ln_final = RMSNorm(d_model)
        else:
            self.ln_final = nn.Identity()

        self.lm_head = Linear(d_model, vocab_size)

        logger.info(
            f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M"
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.lm_head.weight.numel()
        return n_params

    def forward(
        self, x: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, " ... sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x), None

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits, _ = self.forward(x)
            next_token_logits = logits[:, -1]
            temperature_scaled_next_token_logits = next_token_logits / temperature
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = softmax(
                temperature_scaled_next_token_logits, dim=-1
            )
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids
