# Clinical LLM Guardrails via Conditional Latent Routing (CLR) Pipeline

> **Hallucination Mitigation for Medical Language Models through Dual-Lane Inference with Epistemic Uncertainty Intervention**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [System Requirements \& Hardware](#system-requirements--hardware)
- [Installation \& Dependencies](#installation--dependencies)
- [Project Architecture \& Setup](#project-architecture--setup)
- [How to Run](#how-to-run)
- [Benchmarking \& Evaluation Summary](#benchmarking--evaluation-summary)
- [Citation](#citation)
- [License](#license)

---

## Overview

The **Conditional Latent Routing (CLR)** pipeline is a training-time and inference-time framework designed to mitigate clinical hallucinations in large language models without incurring measurable performance degradation on clean inputs — a property we refer to as **zero alignment tax**.

The core idea is a **dual-lane routing architecture**:

| Lane                | Trigger                   | Mechanism                                                              | Goal                                                   |
| ------------------- | ------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------ |
| **Fast Lane** | Clean clinical input      | Encoder latent`z` → Soft-prompt projection → BioMistral generation | Preserve diagnostic accuracy and throughput            |
| **Slow Lane** | Noisy / adversarial input | Raw text + strict system prompt → BioMistral generation               | Refuse hallucinated diagnoses; respond`INCONCLUSIVE` |

### Pipeline Flow

```
                       ┌───────────────────┐
  Patient Record ────→ │ Bio_ClinicalBERT  │
                       │   (Encoder)       │
                       └────────┬──────────┘
                                │
                      ┌─────────▼──────────┐
                      │  Noise Classifier  │
                      │  P(noisy) ≷ 0.5    │
                      └──┬─────────────┬───┘
                         │             │
                    Clean│             │ Noisy
                         ▼             ▼
               ┌──────────────┐ ┌──────────────────┐
               │  FAST LANE   │ │    SLOW LANE     │
               │              │ │                  │
               │ z → Project  │ │ System Prompt +  │
               │ → 32 Soft    │ │ Raw Text →       │
               │   Tokens     │ │ BioMistral       │
               │ + Hard Toks  │ │                  │
               │ → BioMistral │ │ → INCONCLUSIVE   │
               └──────────────┘ └──────────────────┘
```

**Key innovations:**

1. **Conditional Latent Routing** — A Bio_ClinicalBERT encoder simultaneously produces (a) a binary clean/noisy classification and (b) a 768-d latent vector `z`, enabling data-driven lane selection at inference time.
2. **Latent-to-Soft-Prompt Projection** — The latent `z` is projected into 32 virtual token embeddings in BioMistral's 4096-d space, bypassing raw-text injection entirely for the Fast Lane.
3. **T-Adaptive Attention (Epistemic Uncertainty Intervention)** — A zero-overhead monkey patch at the `lm_head` layer that scales logits inversely with hidden-state variance, suppressing high-entropy (uncertain) token generations.
4. **RAGAS Faithfulness via LLM-as-a-Judge** — Two-stage claim extraction → entailment verification using Groq-hosted Llama-4-Scout-17B-16E-Instruct for automated evaluation.

---

## System Requirements & Hardware

### Compute Environment

| Component               | Specification                                                  |
| ----------------------- | -------------------------------------------------------------- |
| **GPU**           | NVIDIA Tesla T4 (16 GB VRAM) — Kaggle environment             |
| **Quantization**  | 4-bit NF4 via`bitsandbytes` with double quantization enabled |
| **Compute dtype** | `torch.float16`                                              |
| **CUDA**          | CUDA 12.x (Kaggle default)                                     |
| **Python**        | ≥ 3.10                                                        |

### Memory Budget

The 4-bit NF4 quantization compresses BioMistral-7B from ~14 GB (FP16) to ~4 GB, enabling single-T4 deployment. The full pipeline (Encoder + Projection + Quantized LLM) fits within the 16 GB VRAM envelope with approximately 3-4 GB headroom for activation memory.

### External API

| Service              | Model                                         | Purpose                                       |
| -------------------- | --------------------------------------------- | --------------------------------------------- |
| **Groq Cloud** | `meta-llama/llama-4-scout-17b-16e-instruct` | LLM-as-a-Judge for RAGAS Faithfulness scoring |

A valid `GROQ_API_KEY` is required for evaluation. Training does not require API access.

---

## Installation & Dependencies

### Core Dependencies

```bash
pip install torch>=2.1.0
pip install transformers>=4.36.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.25.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.66.0
pip install numpy>=1.24.0
pip install groq>=0.4.0
```

Or install all at once from the provided requirements file:

```bash
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` requires a CUDA-capable GPU. On Kaggle, this is pre-installed. For local setups, ensure your CUDA toolkit version matches the `bitsandbytes` build.

### Pre-trained Model Downloads (Automatic)

The following models are downloaded automatically from Hugging Face on first run:

| Model                               | Source                                                              | Size          |
| ----------------------------------- | ------------------------------------------------------------------- | ------------- |
| `emilyalsentzer/Bio_ClinicalBERT` | [Hugging Face](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | ~440 MB       |
| `BioMistral/BioMistral-7B`        | [Hugging Face](https://huggingface.co/BioMistral/BioMistral-7B)        | ~4 GB (4-bit) |

---

## Project Architecture & Setup

### Repository Structure

```
clr/
├── config.py                 # Centralized hyperparameters, paths, model IDs, BnB config
├── encoder.py                # ClinicalLatentEncoder (Bio_ClinicalBERT + classification head)
├── projection.py             # LatentProjection (768-d → 32 × 4096-d soft tokens)
├── inference_pipeline.py     # CLRPipeline routing logic + T-Adaptive Attention patch
├── train.py                  # Training loop (Encoder + Projection; LLM frozen)
├── data_loader.py            # MedicalPairDataset + stratified DataLoader factory
├── evaluate.py               # Full evaluation: routing accuracy, RAGAS, throughput
├── requirements.txt          # Python dependencies
└── paired_dataset_final.json # Paired clinical dataset (200 cases × 2 variants)
 
```

### Kaggle Dataset Setup

On Kaggle, the pipeline expects the dataset at the following path:

```
/kaggle/input/datasets/narendrabayutama/clr-medical-data/paired_dataset_final.json
```

This dataset contains **200 paired clinical cases**, each with:

| Field                 | Description                                       |
| --------------------- | ------------------------------------------------- |
| `id`                | Case identifier (`CASE_001` … `CASE_200`)    |
| `medical_specialty` | Clinical specialty category                       |
| `dataset_A_clear`   | Clean clinical record (Dataset A)                 |
| `dataset_B_noisy`   | Noisy / adversarial clinical record (Dataset B)   |
| `ground_truth_A`    | Reference diagnosis for clean input               |
| `ground_truth_B`    | Expected`INCONCLUSIVE` response for noisy input |

### Training Flow

```
Raw Text → Bio_ClinicalBERT Encoder → (noise_logits, z)
                                          │
                              z → LatentProjection → soft_tokens (32 × 4096)
                                          │
                              [BOS] + soft_tokens + ground_truth_embeds → BioMistral (frozen)
                                          │
                              Loss = 0.3 × CrossEntropy_cls + 0.7 × CrossEntropy_gen
```

Only the **Encoder** and **Projection** parameters are trained. BioMistral-7B remains fully frozen.

---

## How to Run

### 1. Set Environment Variable

```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

### 2. Training (Kaggle Notebook or Local GPU)

```bash
# Full training (10 epochs, gradient accumulation = 8)
python train.py

# Overfit sanity check (4 samples × 50 epochs)
python train.py --overfit_test
```

Training produces checkpoints in `/kaggle/working/checkpoints/` and final model weights in `/kaggle/working/final_models/`:

- `encoder_final.pt` — Trained ClinicalLatentEncoder state dict
- `projection_final.pt` — Trained LatentProjection state dict

### 3. Evaluation

```bash
# Full evaluation with RAGAS Faithfulness (requires GROQ_API_KEY)
python evaluate.py

# Skip RAGAS (no API calls)
python evaluate.py --no_ragas

# Evaluate a subset of cases
python evaluate.py --max_samples 50
```

The evaluation pipeline:

1. Loads the trained encoder and projection weights from `final_models/`.
2. Routes each of the 200 cases through both Dataset A (clean) and Dataset B (noisy).
3. Computes routing accuracy, per-lane latency, throughput, and RAGAS Faithfulness.
4. Writes the complete results to `evaluation_results.json`.

### 4. Inference (Single Case)

```bash
python inference_pipeline.py
```

This runs a 5-case demo showing both Fast Lane and Slow Lane routing with latency and token counts.

---

## Benchmarking & Evaluation Summary

All experiments were conducted on a **Kaggle Notebook with a single NVIDIA Tesla T4 GPU (16 GB VRAM)**. BioMistral-7B was loaded in 4-bit NF4 quantization. The external judge model (`meta-llama/llama-4-scout-17b-16e-instruct`) was accessed via the Groq API.

### Routing Performance (Consistent Across All Trials)

| Metric                      | Value                                              |
| --------------------------- | -------------------------------------------------- |
| **Routing Accuracy**  | **100.00%** (400/400 decisions)              |
| **Alignment Tax**     | **0.00%** (zero degradation on clean inputs) |
| **Baseline Accuracy** | 99.0% (Wibisono, 2026)                             |

### Detailed Results Across Experimental Configurations

#### Dataset A — Clean Clinical Records (Fast Lane)

| Trial                         |    Samples    |   Fast Lane   |    Accuracy    |         Avg Latency |            Throughput | RAGAS Faithfulness | Faith. Samples |
| :---------------------------- | :-----------: | :-----------: | :------------: | ------------------: | --------------------: | -----------------: | :------------: |
| Baseline (1st)                |      200      |      200      |      100%      |           14,491 ms |           10.93 tok/s |              0.291 |      176      |
| Baseline (2nd)                |      200      |      200      |      100%      |           14,901 ms |           10.18 tok/s |              0.298 |      190      |
| Epistemic (greedy)            |      200      |      200      |      100%      |            8,141 ms |           10.51 tok/s |              0.644 |      147      |
| Epistemic (non-greedy)        |      200      |      200      |      100%      |            7,832 ms |           10.67 tok/s |              0.585 |      141      |
| Aleatoric                     |      200      |      200      |      100%      |            8,870 ms |            9.63 tok/s |              0.649 |      146      |
| **Hybrid (T-Adaptive)** | **200** | **200** | **100%** | **13,119 ms** | **10.03 tok/s** |    **0.536** | **146** |

#### Dataset B — Noisy Clinical Records (Slow Lane)

| Trial                         |    Samples    |   Slow Lane   | Routing Accuracy |         Avg Latency |            Throughput |
| :---------------------------- | :-----------: | :-----------: | :--------------: | ------------------: | --------------------: |
| Baseline (1st)                |      200      |      200      |       100%       |           10,440 ms |           10.51 tok/s |
| Baseline (2nd)                |      200      |      200      |       100%       |           11,160 ms |            9.83 tok/s |
| Epistemic (greedy)            |      200      |      200      |       100%       |            9,085 ms |           11.95 tok/s |
| Epistemic (non-greedy)        |      200      |      200      |       100%       |            9,512 ms |           12.46 tok/s |
| Aleatoric                     |      200      |      200      |       100%       |           10,340 ms |           10.61 tok/s |
| **Hybrid (T-Adaptive)** | **200** | **200** |  **100%**  | **10,418 ms** | **10.53 tok/s** |

### Key Findings

1. **Perfect Routing Fidelity** — The ClinicalLatentEncoder achieves 100% routing accuracy across all 1,200 routing decisions (6 trials × 200 cases × 2 datasets), with zero false-positive noisy classifications on clean data and zero false-negative clean classifications on noisy data.
2. **Zero Alignment Tax** — Inserting the CLR guardrail produces 0.00% accuracy loss on clean clinical inputs relative to the unguarded baseline, validating the "zero-tax" hypothesis.
3. **Epistemic Intervention Improves Faithfulness** — The T-Adaptive Attention patch (epistemic variance-based logit scaling) improves RAGAS Faithfulness from 0.291 (baseline) to 0.644 (epistemic greedy), a **+121% relative improvement**, by suppressing high-uncertainty token generations.
4. **Latency-Faithfulness Trade-off** — Epistemic configurations achieve lower latency (7.8–8.9s vs 13.1–14.9s) alongside higher faithfulness, suggesting that uncertainty-aware generation converges to shorter, more factually grounded outputs.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{bayutama2026clr,
  title   = {Clinical LLM Guardrails via Conditional Latent Routing},
  author  = {Wibisono, Narendra Bayutama},
  year    = {2026},
  url     = {https://doi.org/10.5281/zenodo.20919464},
  note    = {Kaggle Notebook with NVIDIA Tesla T4, BioMistral-7B (4-bit NF4)}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with 🧠 Bio_ClinicalBERT · 🦙 BioMistral-7B · ⚡ Groq · 🔬 RAGAS</sub>
</p>
