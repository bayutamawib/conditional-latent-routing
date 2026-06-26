"""
config.py — Centralized configuration for the CLR pipeline.

All paths, hyperparameters, model IDs, and quantization settings
are defined here for consistency across all modules.

Target Environment: Kaggle Dual T4 GPUs (16 GB VRAM each).
"""

import os
import torch
import random
import numpy as np

# ──────────────────────────────────────────────
# Paths (Kaggle)
# ──────────────────────────────────────────────
DATA_DIR = "/kaggle/input/datasets/narendrabayutama/clr-medical-data"
OUTPUT_DIR = "/kaggle/working"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

DATASET_FILE = "paired_dataset_final.json"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILE)

# ──────────────────────────────────────────────
# Model IDs
# ──────────────────────────────────────────────
ENCODER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
LLM_MODEL = "BioMistral/BioMistral-7B"

# ──────────────────────────────────────────────
# Dimensions
# ──────────────────────────────────────────────
ENCODER_DIM = 768       # ClinicalBERT hidden size
LLM_DIM = 4096          # BioMistral-7B embedding dim
NUM_SOFT_TOKENS = 32    # Virtual tokens for soft-prompting

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 1                      # Low to prevent OOM on T4
GRADIENT_ACCUMULATION_STEPS = 8     # Effective batch size = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MAX_SEQ_LEN_ENCODER = 512           # ClinicalBERT max tokens
MAX_SEQ_LEN_LLM = 256               # Max generation length for BioMistral
NOISE_THRESHOLD = 0.5               # Routing decision threshold

# Loss weights (classification vs generation)
LOSS_WEIGHT_CLS = 0.3
LOSS_WEIGHT_GEN = 0.7

# Overfit test settings
OVERFIT_NUM_SAMPLES = 4
OVERFIT_EPOCHS = 50

# ──────────────────────────────────────────────
# Data Split
# ──────────────────────────────────────────────
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# ──────────────────────────────────────────────
# BitsAndBytes 4-bit NF4 Quantization Config
# ──────────────────────────────────────────────
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
}

# ──────────────────────────────────────────────
# Slow Lane System Prompt
# ──────────────────────────────────────────────
SLOW_LANE_SYSTEM_PROMPT = (
    "You are a cautious medical AI assistant. The following patient record "
    "contains noisy, incomplete, or contradictory information. You MUST NOT "
    "attempt a definitive diagnosis. Instead, respond with 'INCONCLUSIVE' "
    "followed by a detailed explanation of why the information is insufficient "
    "and what additional clinical data would be required."
)

# ──────────────────────────────────────────────
# Evaluation — Together AI (LLM-as-a-Judge)
# ──────────────────────────────────────────────
JUDGE_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
