"""
run_ablations.py — Ablation experiment runner for CS336 Assignment 1.

Runs 3 ablation studies comparing Transformer variants:

  Ablation 1 — LayerNorm:
    - Pre-norm RMSNorm (base)
    - No RMSNorm (with LR fallback if loss diverges)
    - Post-norm RMSNorm

  Ablation 2 — Position Embeddings:
    - RoPE (base)
    - NoPE (no positional encoding)

  Ablation 3 — FFN Activations:
    - SwiGLU (base, d_ff = 8/3 * d_model)
    - SiLU (d_ff = 4 * d_model, matches param count)

Each run logs to wandb with `group` and `tags` for easy dashboard comparison.
The training loop is a self-contained adaptation of basics.trainer_model.train()
with wandb logging added — no modifications to the basics/ package required.

Usage:
    python scripts/run_ablations.py  # runs ALL ablation scenarios

    # Run a single ablation group:
    python scripts/run_ablations.py ablation_group=layer_norm
    python scripts/run_ablations.py ablation_group=pos_emb
    python scripts/run_ablations.py ablation_group=ffn
"""

from __future__ import annotations

import os
import sys
import pathlib
import logging
from dataclasses import dataclass, field

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add project root so we can import basics.*
base_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from basics.trainer_model import (
    get_memmap_dataset,
    get_batch,
    memmap_val_iterator,
    _to_device_and_compile,
)
from basics.trainer_utils import (
    run_get_lr_cosine_schedule,
    run_save_checkpoint,
    run_load_checkpoint,
    run_gradient_clipping,
)

# Ablation-specific model — scripts/ablation is not a full package,
# so we add its parent to sys.path and import directly.
sys.path.insert(0, str(base_dir / "scripts"))
from ablation.transformer_variants import AblationTransformerLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training loop (self-contained, with wandb)
# ---------------------------------------------------------------------------

def train_ablation(model: torch.nn.Module, device: torch.device, training_cfg: DictConfig) -> None:
    """Train loop adapted from basics.trainer_model.train() with wandb logging."""
    import wandb

    os.makedirs(training_cfg.save_path, exist_ok=True)

    train_data = get_memmap_dataset(training_cfg.train_data_path)
    val_data = get_memmap_dataset(training_cfg.val_data_path)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )

    start_iter = 0
    if training_cfg.get("resume_checkpoint"):
        ckpt_path = pathlib.Path(training_cfg.save_path) / f"ckpt_iter{training_cfg.resume_checkpoint}.pt"
        start_iter = run_load_checkpoint(ckpt_path, model, optimizer)
        logger.info(f"Resumed at iteration {start_iter} from {ckpt_path}")

    pbar = tqdm(range(start_iter, training_cfg.train_steps), desc="Training", leave=False)
    for iteration in pbar:
        model.train()

        x, y = get_batch(train_data, training_cfg.batch_size, training_cfg.context_length)
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
        )

        # Divergence check — return NaN so caller can retry with lower LR
        if torch.isnan(loss) or torch.isinf(loss):
            wandb.log({"train/loss": float("nan"), "train/lr": 0.0}, step=iteration)
            raise RuntimeError(f"Loss diverged (NaN/Inf) at iteration {iteration}")

        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), training_cfg.clip_grad_norm)

        lr = run_get_lr_cosine_schedule(
            iteration,
            training_cfg.lr,
            training_cfg.min_lr,
            training_cfg.warmup_iters,
            training_cfg.cosine_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        pbar.set_postfix(loss=loss.item(), lr=lr)
        wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=iteration)

        # Validation
        if (iteration + 1) % training_cfg.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in memmap_val_iterator(
                    val_data, training_cfg.batch_size, training_cfg.context_length
                ):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits, _ = model(x_val)
                    val_loss = F.cross_entropy(
                        val_logits.reshape(-1, val_logits.shape[-1]),
                        y_val.reshape(-1),
                    )
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= training_cfg.val_batches:
                        break
                val_loss_mean = float(np.mean(val_losses))
                wandb.log({"val/loss": val_loss_mean}, step=iteration)
                logger.info(f"iter {iteration + 1:05d}: VALID loss = {val_loss_mean:.4f}")

        # Checkpoint
        if (iteration + 1) % training_cfg.save_interval == 0:
            ckpt_name = os.path.join(training_cfg.save_path, f"ckpt_iter{iteration + 1}.pt")
            run_save_checkpoint(model, optimizer, iteration + 1, ckpt_name)


# ---------------------------------------------------------------------------
# Ablation run definitions
# ---------------------------------------------------------------------------

@dataclass
class AblationRun:
    """A single ablation run configuration."""
    name: str
    tags: list = field(default_factory=list)
    model_overrides: dict = field(default_factory=dict)
    training_overrides: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="configs/", config_name="ablation", version_base=None)
def main(cfg: DictConfig) -> None:
    import wandb

    model_cfg = cfg.model
    training_cfg = cfg.training

    # --- Build the list of runs per ablation group ---
    ablation_groups: dict[str, list[AblationRun]] = {
        "layer_norm": [
            AblationRun(
                name="pre-norm-rmsnorm",
                tags=["baseline", "rmsnorm", "pre-norm"],
                model_overrides={"use_rmsnorm": True, "norm_style": "pre"},
            ),
            AblationRun(
                name="no-rmsnorm",
                tags=["no-norm", "pre-norm"],
                model_overrides={"use_rmsnorm": False, "norm_style": "pre"},
                training_overrides={"lr": 0.0001},  # smaller LR to avoid divergence
            ),
            AblationRun(
                name="post-norm-rmsnorm",
                tags=["rmsnorm", "post-norm"],
                model_overrides={"use_rmsnorm": True, "norm_style": "post"},
            ),
        ],
        "pos_emb": [
            AblationRun(
                name="rope",
                tags=["baseline", "rope"],
                model_overrides={"pos_emb": "rope"},
            ),
            AblationRun(
                name="nope",
                tags=["nope"],
                model_overrides={"pos_emb": "nope"},
            ),
        ],
        "ffn": [
            AblationRun(
                name="swiglu",
                tags=["baseline", "swiglu"],
                model_overrides={"ffn_type": "swiglu"},
            ),
            AblationRun(
                name="silu",
                tags=["silu"],
                model_overrides={"ffn_type": "silu"},
            ),
        ],
    }

    # Determine which group(s) to run
    group_name = cfg.get("ablation_group", None)
    if group_name and group_name != "all":
        groups_to_run = {group_name: ablation_groups[group_name]}
    else:
        groups_to_run = ablation_groups

    # --- Run each ablation ---
    for group, runs in groups_to_run.items():
        logger.info(f"\n{'='*60}\n  Ablation: {group}\n{'='*60}")

        for run in runs:
            logger.info(f"\n--- {run.name} ---")

            # Merge run-specific overrides into config
            merged_model = OmegaConf.merge(model_cfg, run.model_overrides)
            merged_training = OmegaConf.merge(training_cfg, run.training_overrides)

            # Set d_ff for SiLU (parameter-matched: d_ff = 4 * d_model)
            if merged_model.get("ffn_type") == "silu":
                merged_model.d_ff = 4 * merged_model.d_model

            run_name = f"{group}/{run.name}"

            # Dynamic model instantiation
            model_kwargs = {
                k: v for k, v in merged_model.items()
                if k in (
                    "vocab_size", "context_length", "d_model",
                    "num_layers", "num_heads", "d_ff", "rope_theta",
                    "use_rmsnorm", "norm_style", "pos_emb", "ffn_type",
                )
            }

            # --- No-RMSNorm: try default LR, then fall back to smaller LR ---
            if run.name == "no-rmsnorm":
                for attempt, (lr_candidate, label) in enumerate([
                    (0.0005, "default_lr"),
                    (0.0001, "small_lr"),
                ]):
                    merged_training.lr = lr_candidate
                    logger.info(
                        f"  No-RMSNorm attempt {attempt + 1}: LR={lr_candidate} ({label})"
                    )
                    wandb.init(
                        project=cfg.wandb.project,
                        entity=cfg.wandb.entity or None,
                        group=group,
                        name=f"{run_name}_lr{lr_candidate}",
                        tags=run.tags + [label],
                        config=OmegaConf.to_container(merged_model, resolve=True),
                        reinit=True,
                    )
                    wandb.config.update({"lr": lr_candidate}, allow_val_change=True)

                    model, device = _to_device_and_compile(
                        AblationTransformerLM(**model_kwargs)
                    )
                    try:
                        train_ablation(model, device, merged_training)
                        break  # succeeded — skip fallback
                    except RuntimeError as e:
                        if "diverged" in str(e).lower() or "nan" in str(e).lower():
                            logger.warning(
                                f"  Diverged with LR={lr_candidate}. "
                                f"Trying smaller LR..."
                            )
                        else:
                            raise
                    finally:
                        wandb.finish(quiet=True)
                else:
                    logger.error("  No-RMSNorm diverged with both LRs. Skipping.")
                continue  # wandb already finished inside the loop

            # --- All other runs: standard init → train → finish ---
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity or None,
                group=group,
                name=run_name,
                tags=run.tags,
                config=OmegaConf.to_container(merged_model, resolve=True),
                reinit=True,
            )
            model = AblationTransformerLM(**model_kwargs)
            model, device = _to_device_and_compile(model)
            train_ablation(model, device, merged_training)
            wandb.finish(quiet=True)

    logger.info("\nAll ablation runs complete.")


if __name__ == "__main__":
    main()
