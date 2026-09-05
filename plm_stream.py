"""
PLM sequence stream for EvoPep (manuscript 2.8):
"a PLM evolutionary encoder built on the ESM-2 transformer architecture and
trained via masked language modeling (MLM; 15% mask ratio) on the peptide corpus"

Provides:
  - MLM fine-tuning of ESM-2 on the ChEMBL peptide corpus
  - mean-pooled sequence embeddings (for QSAR)
  - position-specific amino-acid probability matrices (for dual-modal decoding)
  - naturalness (mean per-position log-likelihood of the sequence)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "esm2_cache")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA)}


class PLMStream:
    def __init__(self, device=None, finetuned_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.finetuned_path = finetuned_path or os.path.join(HERE, "esm2_peptide_mlm.pt")
        if os.path.exists(self.finetuned_path):
            self.model.load_state_dict(torch.load(self.finetuned_path, map_location="cpu"))
            print(f"[PLM] loaded fine-tuned MLM weights: {self.finetuned_path}")
        else:
            print("[PLM] using pretrained ESM-2 weights (no fine-tuned file found)")
        self.model.to(self.device).eval()
        # token ids of the 20 standard amino acids, aligned with AA index
        self.aa_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(a) for a in AA], device=self.device
        )

    # === PART 2 BELOW ===
