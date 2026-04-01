"""
src/training/config.py

All hyperparameters in one place.
Change values here — do not hardcode them in trainer.py or other files.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:

    # ── paths ──────────────────────────────────────────────────────────────────
    data_dir:     str = "data/processed"
    output_dir:   str = "models"
    log_dir:      str = "outputs"

    # ── model architecture ─────────────────────────────────────────────────────
    # Conv-TasNet
    N: int = 256          # encoder filters      (paper: 512 — reduce if OOM)
    L: int = 16           # encoder kernel size
    B: int = 128          # TCN bottleneck channels
    H: int = 256          # TCN hidden channels  (paper: 512 — reduce if OOM)
    P: int = 3            # TCN kernel size
    X: int = 8            # TCN layers per repeat
    R: int = 3            # TCN repeats
    C: int = 3            # max speakers

    # EEND
    d_model:   int = 256  # transformer hidden dim
    n_heads:   int = 4    # attention heads
    n_layers:  int = 4    # transformer layers
    d_ff:      int = 1024 # feed-forward dim
    dropout:   float = 0.1
    subsample: int = 8    # temporal subsampling factor

    # ── data loading ───────────────────────────────────────────────────────────
    chunk_duration: float = 4.0   # seconds per training chunk
    batch_size:     int   = 16     # start here; increase to 8 if MPS allows
    num_workers:    int   = 0     # 0 = safe on MPS

    # ── training ───────────────────────────────────────────────────────────────
    epochs:       int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-5
    grad_clip:    float = 5.0     # max gradient norm (prevents exploding grads)

    # early stopping
    patience:     int   = 5       # stop if no val improvement for N epochs
    lr_patience:  int   = 3       # halve LR after N epochs no improvement

    # ── loss weights (Equation 19 from paper) ─────────────────────────────────
    lambda_sisnr: float = 1.0
    lambda_diar:  float = 0.2
    lambda_exist: float = 0.2

    # ── logging ────────────────────────────────────────────────────────────────
    log_every:      int = 10    # print loss every N batches
    val_every:      int = 1     # validate every N epochs
    save_every:     int = 5     # save checkpoint every N epochs

    # ── debug / fast runs ──────────────────────────────────────────────────────
    max_train:      int | None = None   # set to e.g. 200 for a quick smoke test
    max_val:        int | None = 100    # cap val samples to speed up val loop


def get_config() -> Config:
    return Config()
