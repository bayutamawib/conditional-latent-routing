"""
train.py — Training loop for CLR (Encoder + Projection, BioMistral frozen).

Key architectural decisions:
  1. BOS Token: <s> embedding is prepended BEFORE soft tokens in inputs_embeds.
  2. Weighted Loss: 0.3 * classification_loss + 0.7 * generation_loss.
  3. Gradient Accumulation: Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS.
  4. Overfit Test: --overfit_test flag for information bottleneck sanity check.
  5. Checkpointing: Every epoch with auto-resume for Kaggle 12hr sessions.

Training flow (clean samples):
  Raw text → Encoder → z (768-d)
                         ↓
              Projection → soft_tokens (32, 4096)
                         ↓
              [BOS_embed] + [soft_tokens] + [ground_truth_embeds] → inputs_embeds
                         ↓
              BioMistral (frozen) → logits
                         ↓
              CrossEntropyLoss vs shifted ground_truth tokens → generation_loss
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import (
    LLM_MODEL,
    ENCODER_DIM,
    LLM_DIM,
    NUM_SOFT_TOKENS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_SEQ_LEN_LLM,
    LOSS_WEIGHT_CLS,
    LOSS_WEIGHT_GEN,
    OVERFIT_NUM_SAMPLES,
    OVERFIT_EPOCHS,
    BNB_CONFIG,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    SEED,
    set_seed,
    ensure_dirs,
)
from encoder import ClinicalLatentEncoder, get_encoder_tokenizer, tokenize_for_encoder
from projection import LatentProjection
from data_loader import get_dataloaders, get_overfit_loader


# ──────────────────────────────────────────────
# LLM Loading (4-bit Quantized, Frozen)
# ──────────────────────────────────────────────

def load_frozen_llm(model_name: str = LLM_MODEL, device_map: str = "auto"):
    """
    Load BioMistral-7B in 4-bit NF4 quantization with all weights frozen.

    Returns:
        model: Frozen quantized LLM.
        tokenizer: LLM tokenizer.
    """
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # BioMistral (Mistral-based) may not have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )

    # Freeze ALL LLM parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────
# Core: Build inputs_embeds with BOS + Soft Tokens
# ──────────────────────────────────────────────

def build_inputs_embeds(
    soft_tokens: torch.Tensor,
    target_ids: torch.Tensor,
    llm_model,
    llm_tokenizer,
) -> tuple:
    """
    Construct inputs_embeds by prepending [BOS] + [soft_tokens] + [target_embeds].

    CRITICAL: BioMistral (Mistral-based) requires the <s> BOS token embedding
    before any content. Without it, the model generates gibberish.

    Args:
        soft_tokens: Projected latent tokens, shape (B, NUM_SOFT_TOKENS, LLM_DIM).
        target_ids: Ground truth tokenized IDs, shape (B, target_seq_len).
        llm_model: Frozen BioMistral model.
        llm_tokenizer: BioMistral tokenizer.

    Returns:
        inputs_embeds: Combined embeddings, shape (B, 1 + NUM_SOFT_TOKENS + target_seq_len, LLM_DIM).
        labels: Target labels for loss computation (shifted), shape matching inputs_embeds seq dim.
        attention_mask: Attention mask, shape matching inputs_embeds seq dim.
    """
    batch_size = soft_tokens.size(0)
    device = soft_tokens.device

    # Get the LLM's embedding layer
    embed_layer = llm_model.get_input_embeddings()

    # 1. BOS token embedding: <s>
    bos_id = torch.tensor(
        [[llm_tokenizer.bos_token_id]] * batch_size,
        dtype=torch.long,
        device=device,
    )
    bos_embed = embed_layer(bos_id)  # (B, 1, 4096)

    # 2. Soft tokens are already in LLM embedding space: (B, 32, 4096)
    # Cast to match embed dtype (float16 for quantized model)
    soft_tokens = soft_tokens.to(dtype=bos_embed.dtype)

    # 3. Target (ground truth) token embeddings
    target_embeds = embed_layer(target_ids)  # (B, target_seq_len, 4096)

    # 4. Concatenate: [BOS] + [soft_tokens] + [target_embeds]
    inputs_embeds = torch.cat(
        [bos_embed, soft_tokens, target_embeds],
        dim=1,
    )  # (B, 1 + 32 + target_seq_len, 4096)

    # 5. Build labels for cross-entropy loss
    # Labels should be -100 (ignore) for BOS and soft token positions,
    # and the actual target_ids for the ground truth positions.
    num_prefix = 1 + soft_tokens.size(1)  # BOS + soft tokens
    ignore_labels = torch.full(
        (batch_size, num_prefix),
        fill_value=-100,
        dtype=torch.long,
        device=device,
    )
    labels = torch.cat([ignore_labels, target_ids], dim=1)

    # 6. Build attention mask (all ones — everything is real content)
    total_len = inputs_embeds.size(1)
    attention_mask = torch.ones(
        (batch_size, total_len),
        dtype=torch.long,
        device=device,
    )

    return inputs_embeds, labels, attention_mask


# ──────────────────────────────────────────────
# Tokenize ground truth for LLM
# ──────────────────────────────────────────────

def tokenize_ground_truth(
    texts: list,
    llm_tokenizer,
    max_length: int = MAX_SEQ_LEN_LLM,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Tokenize ground truth texts for the LLM (without BOS — we prepend it manually).

    Args:
        texts: List of ground truth strings.
        llm_tokenizer: BioMistral tokenizer.
        max_length: Max token length.
        device: Target device.

    Returns:
        input_ids: Tensor of shape (B, seq_len).
    """
    encoded = llm_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,  # No BOS — we add it in build_inputs_embeds
    )
    return encoded["input_ids"].to(device)


# ──────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    encoder: nn.Module,
    projection: nn.Module,
    optimizer,
    scaler,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: str = CHECKPOINT_DIR,
):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "projection_state_dict": projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)
    print(f"  📁 Checkpoint saved: {path}")
    return path


def load_latest_checkpoint(
    encoder: nn.Module,
    projection: nn.Module,
    optimizer,
    scaler,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> int:
    """
    Load the latest checkpoint if available.

    Returns:
        start_epoch: The epoch to resume from (0 if no checkpoint found).
    """
    if not os.path.exists(checkpoint_dir):
        return 0

    checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return 0

    latest = checkpoints[-1]
    print(f"  🔄 Resuming from: {latest}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    projection.load_state_dict(ckpt["projection_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    print(f"  📊 Resumed at epoch {start_epoch} "
          f"(train_loss={ckpt['train_loss']:.4f}, val_loss={ckpt['val_loss']:.4f})")
    return start_epoch


# ──────────────────────────────────────────────
# Training Step
# ──────────────────────────────────────────────

def train_step(
    batch: dict,
    encoder: nn.Module,
    projection: nn.Module,
    llm_model,
    llm_tokenizer,
    encoder_tokenizer,
    cls_criterion: nn.Module,
    device: str,
    use_amp: bool = True,
) -> tuple:
    """
    Single training step for one batch.

    Returns:
        total_loss: Weighted combination of cls + gen loss.
        cls_loss_val: Classification loss (float).
        gen_loss_val: Generation loss (float).
    """
    texts = batch["texts"]
    labels = batch["labels"].to(device)
    ground_truths = batch["ground_truths"]

    # ── Step 1: Encode text → (noise_logits, z) ──
    enc_tokens = tokenize_for_encoder(texts, encoder_tokenizer, device=device)
    noise_logits, z = encoder(**enc_tokens)

    # ── Step 2: Classification loss (all samples) ──
    cls_loss = cls_criterion(noise_logits, labels)

    # ── Step 3: Generation loss (clean samples only, label=0) ──
    clean_mask = (labels == 0)
    gen_loss = torch.tensor(0.0, device=device)

    if clean_mask.any():
        # Get clean samples
        z_clean = z[clean_mask]
        clean_truths = [gt for gt, m in zip(ground_truths, clean_mask.tolist()) if m]

        # Project latent → soft tokens
        soft_tokens = projection(z_clean)  # (N_clean, 32, 4096)

        # Tokenize ground truth for LLM
        target_ids = tokenize_ground_truth(clean_truths, llm_tokenizer, device=device)

        # Build inputs_embeds with BOS prepended
        inputs_embeds, gen_labels, attn_mask = build_inputs_embeds(
            soft_tokens, target_ids, llm_model, llm_tokenizer,
        )

        # Forward through frozen LLM
        with torch.no_grad():
            # We need gradients to flow through inputs_embeds (soft tokens)
            # but NOT through the LLM weights. So we use no_grad only for
            # the LLM's internal parameters.
            pass

        # LLM forward — gradients flow through inputs_embeds back to projection+encoder
        llm_outputs = llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=gen_labels,
        )
        gen_loss = llm_outputs.loss

    # ── Step 4: Weighted total loss ──
    total_loss = (LOSS_WEIGHT_CLS * cls_loss) + (LOSS_WEIGHT_GEN * gen_loss)

    return total_loss, cls_loss.item(), gen_loss.item()


# ──────────────────────────────────────────────
# Validation Step
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(
    val_loader,
    encoder: nn.Module,
    projection: nn.Module,
    llm_model,
    llm_tokenizer,
    encoder_tokenizer,
    cls_criterion: nn.Module,
    device: str,
) -> dict:
    """Run validation and return metrics."""
    encoder.eval()
    projection.eval()

    total_cls_loss = 0.0
    total_gen_loss = 0.0
    total_samples = 0
    correct_cls = 0

    for batch in val_loader:
        texts = batch["texts"]
        labels = batch["labels"].to(device)
        ground_truths = batch["ground_truths"]

        # Encode
        enc_tokens = tokenize_for_encoder(texts, encoder_tokenizer, device=device)
        noise_logits, z = encoder(**enc_tokens)

        # Classification metrics
        cls_loss = cls_criterion(noise_logits, labels)
        preds = noise_logits.argmax(dim=-1)
        correct_cls += (preds == labels).sum().item()
        total_cls_loss += cls_loss.item() * len(texts)

        # Generation loss (clean only)
        clean_mask = (labels == 0)
        if clean_mask.any():
            z_clean = z[clean_mask]
            clean_truths = [gt for gt, m in zip(ground_truths, clean_mask.tolist()) if m]

            soft_tokens = projection(z_clean)
            target_ids = tokenize_ground_truth(clean_truths, llm_tokenizer, device=device)
            inputs_embeds, gen_labels, attn_mask = build_inputs_embeds(
                soft_tokens, target_ids, llm_model, llm_tokenizer,
            )

            llm_outputs = llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=gen_labels,
            )
            total_gen_loss += llm_outputs.loss.item() * clean_mask.sum().item()

        total_samples += len(texts)

    encoder.train()
    projection.train()

    n = max(total_samples, 1)
    return {
        "val_cls_loss": total_cls_loss / n,
        "val_gen_loss": total_gen_loss / max(total_samples // 2, 1),
        "val_cls_accuracy": correct_cls / n,
    }


# ──────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────

def train(args):
    """Main training function."""
    set_seed(SEED)
    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load models ──
    print("\n📦 Loading BioMistral-7B (4-bit NF4, frozen)...")
    llm_model, llm_tokenizer = load_frozen_llm()
    print("   ✓ LLM loaded and frozen.")

    print("\n📦 Loading ClinicalLatentEncoder...")
    encoder = ClinicalLatentEncoder().to(device)
    encoder.train()
    print("   ✓ Encoder loaded.")

    print("\n📦 Loading LatentProjection...")
    projection = LatentProjection().to(device)
    projection.train()
    print("   ✓ Projection loaded.")

    encoder_tokenizer = get_encoder_tokenizer()

    # ── Optimizer & Scaler ──
    trainable_params = list(encoder.parameters()) + list(projection.parameters())
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    scaler = GradScaler() if device == "cuda" else None

    cls_criterion = nn.CrossEntropyLoss()

    # ── Data ──
    if args.overfit_test:
        print("\n🧪 OVERFIT TEST MODE — 4 clean samples × 50 epochs")
        train_loader = get_overfit_loader()
        val_loader = None
        num_epochs = OVERFIT_EPOCHS
        grad_accum = 1  # No accumulation for overfit test
    else:
        print("\n📂 Loading data...")
        train_loader, val_loader = get_dataloaders()
        num_epochs = NUM_EPOCHS
        grad_accum = GRADIENT_ACCUMULATION_STEPS
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches:   {len(val_loader)}")

    # ── Resume from checkpoint ──
    start_epoch = 0
    if not args.overfit_test:
        start_epoch = load_latest_checkpoint(encoder, projection, optimizer, scaler)

    # ── Training ──
    print(f"\n🚀 Training — epochs {start_epoch} → {num_epochs}")
    print(f"   Batch size: {BATCH_SIZE}, Grad accum: {grad_accum}, "
          f"Effective batch: {BATCH_SIZE * grad_accum}")
    print(f"   Loss weights: CLS={LOSS_WEIGHT_CLS}, GEN={LOSS_WEIGHT_GEN}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        encoder.train()
        projection.train()

        running_loss = 0.0
        running_cls = 0.0
        running_gen = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for step, batch in pbar:
            # Mixed precision forward
            if scaler and device == "cuda":
                with autocast():
                    total_loss, cls_l, gen_l = train_step(
                        batch, encoder, projection, llm_model, llm_tokenizer,
                        encoder_tokenizer, cls_criterion, device,
                    )
                    scaled_loss = total_loss / grad_accum

                scaler.scale(scaled_loss).backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                total_loss, cls_l, gen_l = train_step(
                    batch, encoder, projection, llm_model, llm_tokenizer,
                    encoder_tokenizer, cls_criterion, device,
                )
                scaled_loss = total_loss / grad_accum
                scaled_loss.backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += total_loss.item()
            running_cls += cls_l
            running_gen += gen_l

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "cls": f"{cls_l:.4f}",
                "gen": f"{gen_l:.4f}",
            })

        # Epoch summary
        n_steps = len(train_loader)
        avg_loss = running_loss / n_steps
        avg_cls = running_cls / n_steps
        avg_gen = running_gen / n_steps
        elapsed = time.time() - epoch_start

        print(f"\n📊 Epoch {epoch + 1}/{num_epochs} — "
              f"Loss: {avg_loss:.4f} (cls={avg_cls:.4f}, gen={avg_gen:.4f}) — "
              f"Time: {elapsed:.1f}s")

        # Validation
        val_loss = 0.0
        if val_loader is not None:
            val_metrics = validate(
                val_loader, encoder, projection, llm_model, llm_tokenizer,
                encoder_tokenizer, cls_criterion, device,
            )
            val_loss = val_metrics["val_cls_loss"] + val_metrics["val_gen_loss"]
            print(f"   Val — cls_loss: {val_metrics['val_cls_loss']:.4f}, "
                  f"gen_loss: {val_metrics['val_gen_loss']:.4f}, "
                  f"cls_acc: {val_metrics['val_cls_accuracy']:.2%}")

        # Save checkpoint
        if not args.overfit_test:
            save_checkpoint(
                epoch, encoder, projection, optimizer, scaler,
                avg_loss, val_loss,
            )
        elif (epoch + 1) % 10 == 0:
            # For overfit test: print loss every 10 epochs
            print(f"   🧪 Overfit check — gen_loss: {avg_gen:.6f}")

    # ── Save final models ──
    print("\n💾 Saving final models...")
    final_dir = os.path.join(OUTPUT_DIR, "final_models")
    os.makedirs(final_dir, exist_ok=True)

    torch.save(encoder.state_dict(), os.path.join(final_dir, "encoder_final.pt"))
    torch.save(projection.state_dict(), os.path.join(final_dir, "projection_final.pt"))
    print(f"   ✓ Saved to {final_dir}/")

    if args.overfit_test:
        print(f"\n🧪 OVERFIT TEST COMPLETE — Final gen_loss: {avg_gen:.6f}")
        if avg_gen < 0.1:
            print("   ✅ Loss converged! Information bottleneck is FEASIBLE.")
        else:
            print("   ⚠️  Loss did NOT converge. Consider increasing NUM_SOFT_TOKENS.")

    print("\n✅ Training complete!")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLR Training Loop")
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help="Run overfit sanity check (4 samples × 50 epochs).",
    )
    args = parser.parse_args()
    train(args)
