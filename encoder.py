"""
encoder.py — Clinical Latent Encoder (Router & Compressor).

Uses Bio_ClinicalBERT as backbone with two outputs:
  1. Classification Head: Binary clean/noisy prediction (noise_logits)
  2. Latent Extraction: CLS pooled output as 768-d latent vector z

Both heads share the same backbone (implicit multi-task regularization).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from config import ENCODER_MODEL, ENCODER_DIM, MAX_SEQ_LEN_ENCODER


class ClinicalLatentEncoder(nn.Module):
    """
    Encoder that routes and compresses clinical text.

    Architecture:
        Bio_ClinicalBERT → [CLS] pooled output (768-d)
                            ├─→ Classification Head → noise_logits (2-d)
                            └─→ Latent vector z (768-d)

    Args:
        model_name: HuggingFace model ID for the encoder backbone.
        encoder_dim: Hidden dimension of the encoder (768 for ClinicalBERT).
        num_classes: Number of classes for noise classification (2: clean/noisy).
        dropout: Dropout rate for classification head.
    """

    def __init__(
        self,
        model_name: str = ENCODER_MODEL,
        encoder_dim: int = ENCODER_DIM,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Backbone: Bio_ClinicalBERT (unfrozen for fine-tuning)
        self.backbone = AutoModel.from_pretrained(model_name)

        # Classification head: clean (0) vs noisy (1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes),
        )

        self.encoder_dim = encoder_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """
        Forward pass through the encoder.

        Args:
            input_ids: Tokenized input IDs, shape (B, seq_len).
            attention_mask: Attention mask, shape (B, seq_len).

        Returns:
            noise_logits: Classification logits, shape (B, 2).
            z: Latent vector (CLS pooled output), shape (B, 768).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token pooled output → latent vector z
        z = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # Classification head
        noise_logits = self.classifier(z)  # (B, 2)

        return noise_logits, z

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple:
        """
        Inference-time prediction.

        Args:
            input_ids: Tokenized input IDs, shape (B, seq_len).
            attention_mask: Attention mask, shape (B, seq_len).
            threshold: Softmax probability threshold for noisy classification.

        Returns:
            is_noisy: Boolean tensor, shape (B,).
            z: Latent vector, shape (B, 768).
            noise_probs: Softmax probabilities, shape (B, 2).
        """
        self.eval()
        noise_logits, z = self.forward(input_ids, attention_mask)
        noise_probs = torch.softmax(noise_logits, dim=-1)
        is_noisy = noise_probs[:, 1] > threshold

        return is_noisy, z, noise_probs


def get_encoder_tokenizer(model_name: str = ENCODER_MODEL):
    """Load the encoder's tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_for_encoder(
    texts: list,
    tokenizer,
    max_length: int = MAX_SEQ_LEN_ENCODER,
    device: str = "cuda",
) -> dict:
    """
    Tokenize a batch of texts for the encoder.

    Args:
        texts: List of raw text strings.
        tokenizer: Encoder tokenizer.
        max_length: Maximum sequence length.
        device: Target device.

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors.
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ClinicalLatentEncoder...")

    # Use CPU for local testing
    device = "cpu"

    # Initialize
    encoder = ClinicalLatentEncoder().to(device)
    tokenizer = get_encoder_tokenizer()

    # Test input
    test_texts = [
        "Patient presents with chest pain and shortness of breath.",
        "Doc I just feel so bad my head hurts and everything is spinning.",
    ]

    tokens = tokenize_for_encoder(test_texts, tokenizer, device=device)

    # Forward pass
    noise_logits, z = encoder(**tokens)
    print(f"  noise_logits shape: {noise_logits.shape}")  # (2, 2)
    print(f"  z shape:            {z.shape}")              # (2, 768)

    assert noise_logits.shape == (2, 2), f"Expected (2, 2), got {noise_logits.shape}"
    assert z.shape == (2, 768), f"Expected (2, 768), got {z.shape}"

    # Predict
    is_noisy, z_pred, probs = encoder.predict(**tokens)
    print(f"  is_noisy:           {is_noisy.tolist()}")
    print(f"  noise_probs:        {probs.tolist()}")

    print("\n✓ Encoder self-test passed.")
