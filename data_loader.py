"""
data_loader.py — Paired Medical Record DataLoader for CLR.

Ingests a single `paired_dataset_final.json` file and unpacks each case
into two training samples:
  - (dataset_A_clear, label=0, ground_truth_A)  → Clean / Diagnosis
  - (dataset_B_noisy, label=1, ground_truth_B)  → Noisy / Inconclusive

Each case in the JSON has the following structure:
  {
    "id": "CASE_XXX",
    "medical_specialty": "...",
    "dataset_A_clear": "...",
    "dataset_B_noisy": "...",
    "ground_truth_A": "...",
    "ground_truth_B": "...",
    "dokumen_pgvector": "..."
  }
"""

import json
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    DATASET_PATH,
    BATCH_SIZE,
    TRAIN_SPLIT,
    SEED,
    set_seed,
)


class MedicalPairDataset(Dataset):
    """
    Unpacks paired medical records into individual training samples.

    Each JSON case yields TWO samples:
      - Clean sample: (text=dataset_A_clear, label=0, ground_truth=ground_truth_A)
      - Noisy sample: (text=dataset_B_noisy, label=1, ground_truth=ground_truth_B)

    Total samples = 2 × number of cases in JSON.
    """

    def __init__(self, data_path: str = DATASET_PATH):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        self.samples: List[Dict[str, str]] = []

        for case in raw_cases:
            case_id = case.get("id", "UNKNOWN")
            specialty = case.get("medical_specialty", "")

            # Clean sample (Dataset A) — label 0
            self.samples.append({
                "case_id": case_id,
                "specialty": specialty,
                "text": case["dataset_A_clear"],
                "label": 0,
                "ground_truth": case["ground_truth_A"],
            })

            # Noisy sample (Dataset B) — label 1
            self.samples.append({
                "case_id": case_id,
                "specialty": specialty,
                "text": case["dataset_B_noisy"],
                "label": 1,
                "ground_truth": case["ground_truth_B"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        return {
            "case_id": sample["case_id"],
            "specialty": sample["specialty"],
            "text": sample["text"],
            "label": sample["label"],
            "ground_truth": sample["ground_truth"],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """
    Custom collate function for variable-length text.

    Returns a dict with:
      - texts: List[str]
      - labels: Tensor of shape (B,)
      - ground_truths: List[str]
      - case_ids: List[str]
      - specialties: List[str]

    Tokenization is deferred to the training loop so each component
    (Encoder, LLM) can use its own tokenizer.
    """
    return {
        "texts": [item["text"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "ground_truths": [item["ground_truth"] for item in batch],
        "case_ids": [item["case_id"] for item in batch],
        "specialties": [item["specialty"] for item in batch],
    }


def get_dataloaders(
    data_path: str = DATASET_PATH,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT,
    seed: int = SEED,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders from the paired dataset.

    Uses stratified splitting to preserve label balance (50/50 clean/noisy)
    across train and validation sets.

    Args:
        data_path: Path to paired_dataset_final.json
        batch_size: Batch size (default 1 for OOM safety)
        train_split: Fraction for training (default 0.8)
        seed: Random seed for reproducibility
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    set_seed(seed)

    dataset = MedicalPairDataset(data_path)
    labels = [s["label"] for s in dataset.samples]

    # Stratified split preserving clean/noisy balance
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_split,
        random_state=seed,
    )
    train_indices, val_indices = next(splitter.split(range(len(dataset)), labels))

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_overfit_loader(
    data_path: str = DATASET_PATH,
    num_samples: int = 4,
    seed: int = SEED,
) -> DataLoader:
    """
    Build a tiny DataLoader for the overfit sanity check.

    Extracts only `num_samples` CLEAN (label=0) samples for the
    information bottleneck feasibility test.

    Args:
        data_path: Path to paired_dataset_final.json
        num_samples: Number of clean samples to extract
        seed: Random seed

    Returns:
        Overfit DataLoader (no shuffle, batch_size = num_samples)
    """
    set_seed(seed)

    dataset = MedicalPairDataset(data_path)

    # Select only clean samples (label=0) for overfit test
    clean_indices = [
        i for i, s in enumerate(dataset.samples) if s["label"] == 0
    ][:num_samples]

    overfit_subset = Subset(dataset, clean_indices)

    return DataLoader(
        overfit_subset,
        batch_size=num_samples,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Allow passing a custom path for local testing
    path = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH

    print(f"Loading dataset from: {path}")
    dataset = MedicalPairDataset(path)
    print(f"Total samples: {len(dataset)} (expected: 2 × num_cases)")

    # Show first sample
    sample = dataset[0]
    print(f"\n--- Sample 0 ---")
    print(f"  Case ID:    {sample['case_id']}")
    print(f"  Specialty:  {sample['specialty']}")
    print(f"  Label:      {sample['label']} ({'clean' if sample['label'] == 0 else 'noisy'})")
    print(f"  Text:       {sample['text'][:120]}...")
    print(f"  Truth:      {sample['ground_truth'][:120]}...")

    # Test dataloaders
    train_loader, val_loader = get_dataloaders(data_path=path)
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"\nBatch keys:    {list(batch.keys())}")
    print(f"Labels shape:  {batch['labels'].shape}")
    print(f"Num texts:     {len(batch['texts'])}")

    # Test overfit loader
    overfit_loader = get_overfit_loader(data_path=path)
    overfit_batch = next(iter(overfit_loader))
    print(f"\nOverfit batch labels: {overfit_batch['labels'].tolist()}")
    print(f"Overfit batch size:   {len(overfit_batch['texts'])}")
    print("\n✓ All data_loader checks passed.")
