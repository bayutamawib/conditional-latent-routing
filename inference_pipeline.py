"""
inference_pipeline.py — Dynamic CLR Routing Pipeline with T-Adaptive Attention.

Implements the core If/Else routing logic:
  - Fast Lane (Clean): Encoder → z → Projection → [BOS + soft_tokens + \n\n + hard_tokens] → BioMistral
  - Slow Lane (Noisy): Encoder detects noise → Raw text + system prompt → BioMistral

Includes Epistemic Uncertainty intervention via Monkey Patching (T-Adaptive Attention).
"""

import time
import types
import torch.nn.functional as F
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    LLM_MODEL,
    LLM_DIM,
    NUM_SOFT_TOKENS,
    MAX_SEQ_LEN_LLM,
    NOISE_THRESHOLD,
    BNB_CONFIG,
    SLOW_LANE_SYSTEM_PROMPT,
    OUTPUT_DIR,
    SEED,
    set_seed,
)
from encoder import ClinicalLatentEncoder, get_encoder_tokenizer, tokenize_for_encoder
from projection import LatentProjection


# ──────────────────────────────────────────────
# T-Adaptive Attention Patch (Epistemic Guardrail)
# ──────────────────────────────────────────────

def apply_t_adaptive_patch(model, alpha=1.0):
    """
    Bedah Saraf V3: Intervensi di level `lm_head` (Zero-Overhead).
    Menghitung Epistemic Uncertainty tanpa merusak memori atau bikin lemot.
    """
    print(f"🧠 Menyuntikkan T-Adaptive Attention V3 (lm_head level, alpha={alpha})...")
    
    # Ambil fungsi bawaan pintu keluar model
    original_lm_head_forward = model.lm_head.forward

    def t_adaptive_lm_head(self, hidden_states):
        # 1. hidden_states di sini sudah merupakan output lapisan TERAKHIR,
        # jadi kita bisa langsung hitung Epistemic Variance-nya.
        variance = torch.var(hidden_states, dim=-1, keepdim=True)
        
        # 2. Rumus T-Adaptive (Suhu ditekan turun jika model bingung)
        adaptive_temp = 1.0 / (1.0 + (alpha * variance))
        adaptive_temp = torch.clamp(adaptive_temp, min=0.1, max=1.0)
        
        # 3. Hitung Logits asli
        logits = original_lm_head_forward(hidden_states)
        
        # 4. Terapkan suhu ke Logits (penawar halusinasi)
        return logits / adaptive_temp

    # Tanamkan fungsi ini hanya di pintu keluar (lm_head)
    model.lm_head.forward = types.MethodType(t_adaptive_lm_head, model.lm_head)
    
    print("   ✓ T-Adaptive Patch V3 (Sangat Cepat) berhasil ditanam!")
    return model


# ──────────────────────────────────────────────
# Pipeline Utama
# ──────────────────────────────────────────────

class CLRPipeline:
    """
    Conditional Latent Routing inference pipeline.

    Routes clinical text through either:
      - Fast Lane: Latent-only soft-prompting (no raw text to LLM)
      - Slow Lane: Full raw text + strict system prompt

    Args:
        encoder: Trained ClinicalLatentEncoder.
        projection: Trained LatentProjection.
        llm_model: Frozen BioMistral-7B (4-bit).
        llm_tokenizer: BioMistral tokenizer.
        encoder_tokenizer: ClinicalBERT tokenizer.
        device: Target device.
        noise_threshold: Softmax threshold for noisy classification.
    """

    def __init__(
        self,
        encoder: ClinicalLatentEncoder,
        projection: LatentProjection,
        llm_model,
        llm_tokenizer,
        encoder_tokenizer,
        device: str = "cuda",
        noise_threshold: float = NOISE_THRESHOLD,
    ):
        self.encoder = encoder.to(device).eval()
        self.projection = projection.to(device).eval()
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.device = device
        self.noise_threshold = noise_threshold

        # Cache BOS embedding
        self._bos_embed = None

    def _get_bos_embed(self) -> torch.Tensor:
        """Get the cached BOS token embedding (1, 1, LLM_DIM)."""
        if self._bos_embed is None:
            embed_layer = self.llm_model.get_input_embeddings()
            bos_id = torch.tensor(
                [[self.llm_tokenizer.bos_token_id]],
                dtype=torch.long,
                device=self.device,
            )
            self._bos_embed = embed_layer(bos_id)
        return self._bos_embed

    @torch.no_grad()
    def route(self, raw_text: str, max_new_tokens: int = MAX_SEQ_LEN_LLM) -> Dict:
        """
        Route a single clinical text through the CLR pipeline.
        """
        start_time = time.time()

        # ── Step 1: Encode → classify + extract latent ──
        enc_tokens = tokenize_for_encoder(
            [raw_text], self.encoder_tokenizer, device=self.device,
        )
        
        # Penawar anti-error token_type_ids
        enc_tokens.pop("token_type_ids", None)
        
        is_noisy, z, noise_probs = self.encoder.predict(
            **enc_tokens, threshold=self.noise_threshold,
        )

        is_noisy_bool = is_noisy[0].item()
        noise_prob = noise_probs[0, 1].item()

        # ── Step 2: Route ──
        if not is_noisy_bool:
            # FAST LANE HYBRID — Latent Soft Tokens + Hard Tokens Teks Asli
            output_text, num_tokens = self._fast_lane(z, raw_text, max_new_tokens)
            lane = "fast"
        else:
            # SLOW LANE — raw text + strict system prompt
            output_text, num_tokens = self._slow_lane(raw_text, max_new_tokens)
            lane = "slow"

        latency_ms = (time.time() - start_time) * 1000

        return {
            "output": output_text,
            "lane": lane,
            "is_noisy": is_noisy_bool,
            "noise_prob": noise_prob,
            "latency_ms": latency_ms,
            "num_tokens": num_tokens,
        }

    def _fast_lane(
        self, z: torch.Tensor, raw_text: str, max_new_tokens: int,
    ) -> tuple:
        """
        Fast Lane (HYBRID V2): Latent + Separator + Hard Tokens + Repetition Penalty.
        Embedding order: [BOS] + [Soft Tokens] + [\n\n] + [Hard Tokens] → BioMistral
        """
        # 1. Project latent → soft tokens (Instruksi Perilaku)
        soft_tokens = self.projection(z)  

        # Get BOS embedding
        bos_embed = self._get_bos_embed()  
        soft_tokens = soft_tokens.to(dtype=bos_embed.dtype)

        # 2. Siapkan Hard Tokens & Separator
        # Tokenizer untuk Separator (Culture Shock Absorber)
        sep_inputs = self.llm_tokenizer(
            "\n\n", 
            return_tensors="pt", 
            add_special_tokens=False
        ).to(self.device)
        
        # Tokenizer untuk Teks Pasien Asli (Jangkar Fakta)
        hard_inputs = self.llm_tokenizer(
            raw_text,
            return_tensors="pt",
            add_special_tokens=False, 
            truncation=True,
            max_length=1024
        ).to(self.device)

        # 3. Ubah KTP Token menjadi Vektor Makna
        embed_layer = self.llm_model.get_input_embeddings()
        sep_embeds = embed_layer(sep_inputs["input_ids"])
        hard_embeds = embed_layer(hard_inputs["input_ids"])

        # 4. JAHIT MATRIKSNYA!
        # Urutan: [BOS] + [128 Soft Tokens] + [\n\n] + [Teks Asli]
        inputs_embeds = torch.cat([bos_embed, soft_tokens, sep_embeds, hard_embeds], dim=1) 

        # Attention mask
        seq_len = inputs_embeds.size(1)
        attn_mask = torch.ones(
            (1, seq_len),
            dtype=torch.long,
            device=self.device,
        )

        # 5. Generate (BioMistral Mengunyah Vektor Gabungan)
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # BUKA KERAN SAMPLING!
            top_p=0.9,               # Mencegah halusinasi token acak ekstrem
            temperature=1.0,         # Suhu dasar (akan dimanipulasi oleh V3)
            repetition_penalty=1.15, 
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )

        # Decode (Ubah kembali ke teks manusia)
        output_text = self.llm_tokenizer.decode(
            outputs[0], skip_special_tokens=True,
        ).strip()

        num_tokens = outputs.shape[1]
        return output_text, num_tokens

    def _slow_lane(
        self, raw_text: str, max_new_tokens: int,
    ) -> tuple:
        """
        Slow Lane: Generate INCONCLUSIVE response from raw text + system prompt.

        Uses standard tokenized input with the strict system prompt prepended.
        """
        # Build prompt
        prompt = (
            f"### System:\n{SLOW_LANE_SYSTEM_PROMPT}\n\n"
            f"### Patient Record:\n{raw_text}\n\n"
            f"### Assessment:\n"
        )

        # Tokenize with BOS (standard tokenization)
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Allow longer input for full raw text
            add_special_tokens=True,  # Includes <s> BOS automatically
        ).to(self.device)

        # Generate
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # BUKA KERAN SAMPLING!
            top_p=0.9,
            temperature=1.0,         # Suhu dasar (akan dimanipulasi oleh V3)
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )

        # Decode only the generated portion (skip the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, prompt_len:]
        output_text = self.llm_tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        ).strip()

        num_tokens = len(generated_ids)
        return output_text, num_tokens

    @torch.no_grad()
    def batch_route(
        self, texts: list, max_new_tokens: int = MAX_SEQ_LEN_LLM,
    ) -> list:
        """
        Route a batch of texts through the pipeline (sequentially).
        """
        results = []
        for text in texts:
            result = self.route(text, max_new_tokens)
            results.append(result)
        return results


# ──────────────────────────────────────────────
# Pipeline Factory
# ──────────────────────────────────────────────

def load_pipeline(
    encoder_path: str,
    projection_path: str,
    llm_model_name: str = LLM_MODEL,
    device: str = "cuda",
) -> CLRPipeline:
    """
    Load a trained CLR pipeline from saved checkpoints.
    """
    print("📦 Loading CLR Pipeline...")

    # Load encoder
    encoder = ClinicalLatentEncoder()
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True)
    )
    encoder = encoder.to(device).eval()
    print("   ✓ Encoder loaded.")

    # Load projection
    projection = LatentProjection()
    projection.load_state_dict(
        torch.load(projection_path, map_location="cpu", weights_only=True)
    )
    projection = projection.to(device).eval()
    print("   ✓ Projection loaded.")

    # Load LLM (4-bit quantized, frozen)
    bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        quantization_config=bnb_config,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    llm_model.eval()
    
    # --- INTERVENSI T-ADAPTIVE ---
    # Memanggil patch Epistemic Uncertainty sebelum masuk ke Pipeline
    llm_model = apply_t_adaptive_patch(llm_model, alpha=1.5)
    
    print("   ✓ BioMistral-7B loaded (4-bit NF4) + T-Adaptive V1.")

    # Encoder tokenizer
    encoder_tokenizer = get_encoder_tokenizer()

    pipeline = CLRPipeline(
        encoder=encoder,
        projection=projection,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        encoder_tokenizer=encoder_tokenizer,
        device=device,
    )
    print("   ✓ Pipeline ready.\n")
    return pipeline


# ──────────────────────────────────────────────
# Quick self-test / demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import json
    from config import OUTPUT_DIR, DATASET_PATH

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths to trained models
    final_dir = os.path.join(OUTPUT_DIR, "final_models")
    encoder_path = os.path.join(final_dir, "encoder_final.pt")
    projection_path = os.path.join(final_dir, "projection_final.pt")

    if not os.path.exists(encoder_path):
        print("❌ No trained models found. Run train.py first.")
        exit(1)

    # Load pipeline
    pipeline = load_pipeline(encoder_path, projection_path, device=device)

    # Load 5 sample cases
    with open(DATASET_PATH, "r") as f:
        cases = json.load(f)[:5]

    print("=" * 70)
    print("CLR INFERENCE DEMO — 5 Sample Cases")
    print("=" * 70)

    for case in cases:
        case_id = case["id"]

        # Test with clean text (should → Fast Lane)
        print(f"\n{'─' * 50}")
        print(f"Case {case_id} — CLEAN INPUT (Dataset A)")
        result_a = pipeline.route(case["dataset_A_clear"])
        print(f"  Lane:       {result_a['lane'].upper()}")
        print(f"  Noise prob: {result_a['noise_prob']:.3f}")
        print(f"  Latency:    {result_a['latency_ms']:.0f} ms")
        print(f"  Tokens:     {result_a['num_tokens']}")
        print(f"  Output:     {result_a['output'][:200]}...")

        # Test with noisy text (should → Slow Lane)
        print(f"\nCase {case_id} — NOISY INPUT (Dataset B)")
        result_b = pipeline.route(case["dataset_B_noisy"])
        print(f"  Lane:       {result_b['lane'].upper()}")
        print(f"  Noise prob: {result_b['noise_prob']:.3f}")
        print(f"  Latency:    {result_b['latency_ms']:.0f} ms")
        print(f"  Tokens:     {result_b['num_tokens']}")
        print(f"  Output:     {result_b['output'][:200]}...")

    print(f"\n{'=' * 70}")
    print("✓ Inference demo complete.")