"""
projection.py — Latent-to-Embedding Projection Layer.

Maps the Encoder's 768-d latent vector z into NUM_SOFT_TOKENS virtual
token embeddings in the LLM's embedding space (4096-d each).

These soft tokens are prepended to BioMistral's inputs_embeds to create
a "soft prompt" that carries the clinical context from the latent vector.
"""

import torch
import torch.nn as nn

from config import ENCODER_DIM, LLM_DIM, NUM_SOFT_TOKENS


class LatentProjection(nn.Module):
    """
    Projects encoder latent vector z into LLM embedding space.

    Architecture:
        z (B, 768) → Linear → (B, 4096 * 32) → Reshape → (B, 32, 4096)

    The output represents NUM_SOFT_TOKENS virtual token embeddings that
    are prepended to BioMistral's input embedding sequence.

    Args:
        encoder_dim: Input dimension (768 for ClinicalBERT).
        llm_dim: Output dimension per token (4096 for BioMistral-7B).
        num_soft_tokens: Number of virtual tokens to generate.
        dropout: Dropout rate before projection.
    """

    def __init__(
        self,
        encoder_dim: int = ENCODER_DIM,
        llm_dim: int = LLM_DIM,
        num_soft_tokens: int = NUM_SOFT_TOKENS,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_soft_tokens = num_soft_tokens
        self.llm_dim = llm_dim

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, llm_dim * num_soft_tokens),
            nn.GELU(),
        )

        # Layer norm for stabilizing projected embeddings
        self.layer_norm = nn.LayerNorm(llm_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project latent vector z into soft token embeddings.

        Args:
            z: Latent vector from encoder, shape (B, encoder_dim).

        Returns:
            soft_tokens: Virtual token embeddings, shape (B, num_soft_tokens, llm_dim).
        """
        batch_size = z.size(0)

        # Project: (B, 768) → (B, 4096 * 32)
        projected = self.projection(z)

        # Reshape: (B, 4096 * 32) → (B, 32, 4096)
        soft_tokens = projected.view(batch_size, self.num_soft_tokens, self.llm_dim)

        # Normalize each token embedding
        soft_tokens = self.layer_norm(soft_tokens)

        return soft_tokens


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from config import ENCODER_DIM, LLM_DIM, NUM_SOFT_TOKENS

    print("Testing LatentProjection...")

    device = "cpu"
    batch_size = 2

    # Simulate encoder output
    z = torch.randn(batch_size, ENCODER_DIM, device=device)

    # Initialize projection
    projection = LatentProjection().to(device)

    # Forward pass
    soft_tokens = projection(z)
    print(f"  Input z shape:      {z.shape}")           # (2, 768)
    print(f"  soft_tokens shape:  {soft_tokens.shape}")  # (2, 32, 4096)

    expected = (batch_size, NUM_SOFT_TOKENS, LLM_DIM)
    assert soft_tokens.shape == expected, f"Expected {expected}, got {soft_tokens.shape}"

    # Check parameters
    total_params = sum(p.numel() for p in projection.parameters())
    trainable = sum(p.numel() for p in projection.parameters() if p.requires_grad)
    print(f"  Total params:       {total_params:,}")
    print(f"  Trainable params:   {trainable:,}")

    print("\n✓ Projection self-test passed.")
