"""
collect_report_data.py
Run from your project root:

    .venv/bin/python3 collect_report_data.py

What this does:
  1. Extracts all info from eend_ss_best.pth
  2. Runs inference on N test samples (default 50)
  3. Computes SI-SDR, DER, speaker counting accuracy
  4. Saves everything to outputs/report_data.json
  5. Prints a formatted summary ready to paste into your IEEE paper

Takes ~15-30 minutes for 50 samples on MPS.
"""

import json
import sys
import time
import random
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from src.models.eend_ss import EENDSS
from src.training.config import get_config

# ── config ────────────────────────────────────────────────────────────────────
N_TEST_SAMPLES  = 50    # increase to 100+ for final paper numbers
CKPT_PATH       = Path("models/eend_ss_best.pth")
DATA_DIR        = Path("data/processed")
OUTPUT_PATH     = Path("outputs/report_data.json")
SAMPLE_RATE     = 16000
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")

Path("outputs").mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def si_sdr_np(estimate: np.ndarray, target: np.ndarray, eps=1e-8) -> float:
    estimate = estimate - estimate.mean()
    target   = target   - target.mean()
    dot      = np.dot(estimate, target)
    tgt_nrg  = np.dot(target, target) + eps
    s_tgt    = (dot / tgt_nrg) * target
    e_noise  = estimate - s_tgt
    return float(10 * np.log10(
        (np.dot(s_tgt, s_tgt) + eps) / (np.dot(e_noise, e_noise) + eps)
    ))


def best_pit_sisnr(estimates: np.ndarray, targets: np.ndarray) -> tuple[float, list]:
    """Try all permutations, return best mean SI-SDR and per-speaker scores."""
    n = min(estimates.shape[0], targets.shape[0])
    T = min(estimates.shape[1], targets.shape[1])
    est = estimates[:n, :T]
    tgt = targets[:n,  :T]

    best_mean  = -999
    best_scores = []
    for perm in permutations(range(n)):
        scores = [si_sdr_np(est[perm[i]], tgt[i]) for i in range(n)]
        mean   = sum(scores) / len(scores)
        if mean > best_mean:
            best_mean   = mean
            best_scores = scores
    return best_mean, best_scores


def compute_der(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Simplified DER from binary frame arrays [T, C]."""
    T  = min(pred.shape[0], ref.shape[0])
    C  = min(pred.shape[1], ref.shape[1])
    pred, ref = pred[:T, :C], ref[:T, :C]

    total_ref = ref.sum()
    if total_ref == 0:
        return {"DER": 0.0, "FA": 0.0, "MISS": 0.0}

    # find best column alignment
    best_fa, best_miss = 1.0, 1.0
    for perm in permutations(range(C)):
        p = pred[:, list(perm)]
        fa   = (p * (1 - ref)).sum() / total_ref
        miss = ((1 - p) * ref).sum()  / total_ref
        if fa + miss < best_fa + best_miss:
            best_fa, best_miss = float(fa), float(miss)

    return {
        "DER":  round(min(best_fa + best_miss, 1.0), 4),
        "FA":   round(best_fa, 4),
        "MISS": round(best_miss, 4),
    }


def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = AF.resample(wav, sr, SAMPLE_RATE)
    return wav


# ── STEP 1: Extract checkpoint info ──────────────────────────────────────────

print("=" * 65)
print("STEP 1: Extracting checkpoint info")
print("=" * 65)

assert CKPT_PATH.exists(), f"Checkpoint not found: {CKPT_PATH}"
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

ckpt_info = {
    "epoch":            ckpt.get("epoch", "unknown"),
    "val_loss":         round(float(ckpt.get("val_loss", 0)), 4),
    "val_sisnr_db":     round(float(ckpt.get("val_sisnr", 0)), 4),
    "checkpoint_path":  str(CKPT_PATH),
}

# extract config that was used during training
saved_cfg = ckpt.get("config", {})
model_config = {
    "N_encoder_filters":      saved_cfg.get("N", "N/A"),
    "L_encoder_kernel":       saved_cfg.get("L", "N/A"),
    "B_bottleneck_channels":  saved_cfg.get("B", "N/A"),
    "H_hidden_channels":      saved_cfg.get("H", "N/A"),
    "X_tcn_layers_per_repeat":saved_cfg.get("X", "N/A"),
    "R_tcn_repeats":          saved_cfg.get("R", "N/A"),
    "d_model_transformer":    saved_cfg.get("d_model", "N/A"),
    "n_heads_attention":      saved_cfg.get("n_heads", "N/A"),
    "n_layers_transformer":   saved_cfg.get("n_layers", "N/A"),
    "max_speakers":           saved_cfg.get("C", 3),
    "chunk_duration_s":       saved_cfg.get("chunk_duration", 4.0),
    "batch_size":             saved_cfg.get("batch_size", "N/A"),
    "learning_rate":          saved_cfg.get("lr", "N/A"),
    "lambda_sisnr":           saved_cfg.get("lambda_sisnr", "N/A"),
    "lambda_diar":            saved_cfg.get("lambda_diar", "N/A"),
    "lambda_exist":           saved_cfg.get("lambda_exist", "N/A"),
}

print(f"  Checkpoint epoch   : {ckpt_info['epoch']}")
print(f"  Val loss           : {ckpt_info['val_loss']}")
print(f"  Val SI-SNR         : {ckpt_info['val_sisnr_db']} dB")
print(f"  Arch config        : N={model_config['N_encoder_filters']}, "
      f"H={model_config['H_hidden_channels']}, "
      f"d_model={model_config['d_model_transformer']}")


# ── STEP 2: Count model parameters ───────────────────────────────────────────

print("\nSTEP 2: Counting parameters")
cfg   = get_config()
model = EENDSS(
    N=cfg.N, L=cfg.L, B=cfg.B, H=cfg.H, P=cfg.P,
    X=cfg.X, R=cfg.R, C=cfg.C, d_model=cfg.d_model,
    n_heads=cfg.n_heads, n_layers=cfg.n_layers,
    d_ff=cfg.d_ff, dropout=cfg.dropout, subsample=cfg.subsample,
)

state = {k: v for k, v in ckpt["model_state"].items() if "pos_enc.pe" not in k}
model.load_state_dict(state, strict=False)
model = model.to(DEVICE)
model.eval()

total_params    = sum(p.numel() for p in model.parameters())
tasnet_params   = sum(p.numel() for p in model.conv_tasnet.parameters())
eend_params     = sum(p.numel() for p in model.eend.parameters())
model_size_mb   = total_params * 4 / 1e6

param_info = {
    "total_parameters":        total_params,
    "conv_tasnet_parameters":  tasnet_params,
    "eend_parameters":         eend_params,
    "model_size_mb":           round(model_size_mb, 1),
}

print(f"  Total parameters   : {total_params:,}")
print(f"  Conv-TasNet        : {tasnet_params:,}")
print(f"  EEND               : {eend_params:,}")
print(f"  Model size         : {model_size_mb:.1f} MB")


# ── STEP 3: Run inference on test samples ─────────────────────────────────────

print(f"\nSTEP 3: Running inference on {N_TEST_SAMPLES} test samples")
print("(this takes ~15-30 minutes on MPS)\n")

test_manifest = DATA_DIR / "test" / "test_manifest.json"
assert test_manifest.exists(), f"Test manifest not found: {test_manifest}"

with open(test_manifest) as f:
    all_samples = json.load(f)

# stratify: equal mix of 2-speaker and 3-speaker samples
two_spk   = [s for s in all_samples if s["num_speakers"] == 2]
three_spk = [s for s in all_samples if s["num_speakers"] == 3]
n_each    = N_TEST_SAMPLES // 2
random.seed(42)
selected  = random.sample(two_spk,   min(n_each, len(two_spk))) + \
            random.sample(three_spk, min(n_each, len(three_spk)))
random.shuffle(selected)

results_2spk = []
results_3spk = []
errors       = []

t0 = time.time()
for i, sample in enumerate(selected):
    mix_name = Path(sample["mixture_path"]).name
    mix_path = DATA_DIR / "test" / "mixtures" / mix_name
    lbl_path = DATA_DIR / "test" / "mixtures" / Path(sample["label_path"]).name

    if not mix_path.exists():
        errors.append(f"Missing: {mix_name}")
        continue

    try:
        # load mixture
        mixture = load_audio(mix_path)
        duration = mixture.shape[-1] / SAMPLE_RATE

        # load ground truth sources
        gt_sources = []
        for sp in sample["source_paths"]:
            sp_path = DATA_DIR / "test" / "mixtures" / Path(sp).name
            if sp_path.exists():
                src = load_audio(sp_path).squeeze(0).numpy()
                gt_sources.append(src)

        # load ground truth labels
        gt_labels = None
        if lbl_path.exists():
            gt_labels = np.load(str(lbl_path))  # [T_frames, C]

        # run model
        with torch.no_grad():
            mixture_in = mixture.unsqueeze(0).to(DEVICE)
            result = model.inference(mixture_in, threshold=0.5)

        separated    = result["separated"].squeeze(0).cpu().numpy()  # [C_hat, T]
        diar_binary  = result["diar_binary"].squeeze(0).cpu().numpy()  # [T_sub, C]
        num_speakers = result["num_speakers"]
        exist_probs  = result["exist_probs"].squeeze(0).cpu().numpy().tolist()

        # ── SI-SDR ──
        sisnr_mean, sisnr_per_spk = None, []
        if gt_sources:
            sisnr_mean, sisnr_per_spk = best_pit_sisnr(
                separated[:num_speakers], np.stack(gt_sources)
            )

        # ── DER ──
        der_result = {}
        if gt_labels is not None:
            T_sub = diar_binary.shape[0]
            scale = T_sub / gt_labels.shape[0]
            from scipy.ndimage import zoom
            gt_resized = (zoom(gt_labels.astype(float), (scale, 1)) > 0.5).astype(float)
            der_result = compute_der(diar_binary, gt_resized)

        # ── speaker count accuracy ──
        spk_correct = (num_speakers == sample["num_speakers"])

        sample_result = {
            "mixture_id":       sample["mixture_id"],
            "duration":         round(duration, 2),
            "true_speakers":    sample["num_speakers"],
            "pred_speakers":    num_speakers,
            "speaker_correct":  spk_correct,
            "sisnr_mean_db":    round(sisnr_mean, 3) if sisnr_mean is not None else None,
            "sisnr_per_speaker":[round(s, 3) for s in sisnr_per_spk],
            "der":              der_result,
            "exist_probs":      [round(p, 3) for p in exist_probs],
        }

        if sample["num_speakers"] == 2:
            results_2spk.append(sample_result)
        else:
            results_3spk.append(sample_result)

        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (len(selected) - i - 1)
        sisnr_str = f"{sisnr_mean:.2f} dB" if sisnr_mean else "N/A"
        der_str   = f"{der_result.get('DER', 0)*100:.1f}%" if der_result else "N/A"
        spk_str   = f"{sample['num_speakers']}→{num_speakers}"
        print(f"  [{i+1:3d}/{len(selected)}] {sample['mixture_id']} | "
              f"spk {spk_str} {'✓' if spk_correct else '✗'} | "
              f"SI-SDR {sisnr_str} | DER {der_str} | ETA {eta:.0f}s")

    except Exception as e:
        errors.append(f"{sample['mixture_id']}: {e}")
        print(f"  [{i+1:3d}] ERROR — {e}")


# ── STEP 4: Aggregate metrics ─────────────────────────────────────────────────

print("\nSTEP 4: Computing aggregate metrics")

def aggregate(results: list) -> dict:
    if not results:
        return {}
    sisnrs  = [r["sisnr_mean_db"] for r in results if r["sisnr_mean_db"] is not None]
    ders    = [r["der"]["DER"]    for r in results if r["der"]]
    fas     = [r["der"]["FA"]     for r in results if r["der"]]
    misses  = [r["der"]["MISS"]   for r in results if r["der"]]
    correct = [r["speaker_correct"] for r in results]
    durs    = [r["duration"] for r in results]

    return {
        "n_samples":              len(results),
        "mean_sisnr_db":          round(float(np.mean(sisnrs)), 3)  if sisnrs  else None,
        "std_sisnr_db":           round(float(np.std(sisnrs)),  3)  if sisnrs  else None,
        "median_sisnr_db":        round(float(np.median(sisnrs)),3) if sisnrs  else None,
        "mean_DER":               round(float(np.mean(ders)),   3)  if ders    else None,
        "mean_FA":                round(float(np.mean(fas)),    3)  if fas     else None,
        "mean_MISS":              round(float(np.mean(misses)), 3)  if misses  else None,
        "speaker_count_accuracy": round(float(np.mean(correct)), 3) if correct else None,
        "mean_duration_s":        round(float(np.mean(durs)),   2)  if durs    else None,
    }

agg_all  = aggregate(results_2spk + results_3spk)
agg_2spk = aggregate(results_2spk)
agg_3spk = aggregate(results_3spk)


# ── STEP 5: Dataset statistics ────────────────────────────────────────────────

print("\nSTEP 5: Collecting dataset statistics")
dataset_stats = {}
for split in ["train", "val", "test"]:
    mf = DATA_DIR / split / f"{split}_manifest.json"
    if mf.exists():
        samples = json.load(open(mf))
        durs  = [s["duration"] for s in samples]
        spk2  = sum(1 for s in samples if s["num_speakers"] == 2)
        spk3  = sum(1 for s in samples if s["num_speakers"] == 3)
        dataset_stats[split] = {
            "n_samples":        len(samples),
            "mean_duration_s":  round(float(np.mean(durs)), 2),
            "min_duration_s":   round(float(np.min(durs)),  2),
            "max_duration_s":   round(float(np.max(durs)),  2),
            "n_2speaker":       spk2,
            "n_3speaker":       spk3,
            "pct_2speaker":     round(spk2 / len(samples) * 100, 1),
            "pct_3speaker":     round(spk3 / len(samples) * 100, 1),
        }
        print(f"  {split}: {len(samples)} samples | "
              f"2-spk: {spk2} | 3-spk: {spk3} | "
              f"avg dur: {np.mean(durs):.1f}s")


# ── STEP 6: Save everything ───────────────────────────────────────────────────

report_data = {
    "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
    "checkpoint":      ckpt_info,
    "model_config":    model_config,
    "model_params":    param_info,
    "dataset_stats":   dataset_stats,
    "eval_config": {
        "n_test_samples":  len(selected),
        "device":          str(DEVICE),
        "threshold":       0.5,
        "random_seed":     42,
    },
    "results": {
        "overall":    agg_all,
        "2_speaker":  agg_2spk,
        "3_speaker":  agg_3spk,
    },
    "per_sample_results": results_2spk + results_3spk,
    "errors": errors,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\nSaved to {OUTPUT_PATH}")


# ── STEP 7: Print paper-ready summary ─────────────────────────────────────────

print("\n" + "=" * 65)
print("PAPER-READY SUMMARY (copy into your IEEE report)")
print("=" * 65)

print(f"""
MODEL ARCHITECTURE
  Encoder filters (N)       : {model_config['N_encoder_filters']}
  Encoder kernel size (L)   : {model_config['L_encoder_kernel']}
  TCN bottleneck (B)        : {model_config['B_bottleneck_channels']}
  TCN hidden (H)            : {model_config['H_hidden_channels']}
  TCN layers×repeats (X×R)  : {model_config['X_tcn_layers_per_repeat']}×{model_config['R_tcn_repeats']}
  Transformer d_model       : {model_config['d_model_transformer']}
  Transformer heads×layers  : {model_config['n_heads_attention']}×{model_config['n_layers_transformer']}
  Total parameters          : {param_info['total_parameters']:,}
  Model size                : {param_info['model_size_mb']} MB

TRAINING
  Best epoch                : {ckpt_info['epoch']}
  Val loss (best)           : {ckpt_info['val_loss']}
  Val SI-SNR (best)         : {ckpt_info['val_sisnr_db']} dB
  Loss weights λ_sep/λ_diar : {model_config['lambda_sisnr']}/{model_config['lambda_diar']}
  Batch size                : {model_config['batch_size']}
  Learning rate             : {model_config['learning_rate']}

DATASET (LibriSpeech mixtures)
  Train / Val / Test        : {dataset_stats.get('train',{}).get('n_samples','?')} / {dataset_stats.get('val',{}).get('n_samples','?')} / {dataset_stats.get('test',{}).get('n_samples','?')}
  Audio duration (avg)      : {dataset_stats.get('test',{}).get('mean_duration_s','?')}s
  Speakers per mixture      : 2–3
  Overlap ratio             : 30–90%

TEST SET RESULTS  (n={agg_all.get('n_samples','?')} samples)
┌─────────────────────┬─────────┬─────────┬─────────┐
│ Metric              │ Overall │ 2-spk   │ 3-spk   │
├─────────────────────┼─────────┼─────────┼─────────┤
│ SI-SDR (dB) ↑      │ {str(agg_all.get('mean_sisnr_db','N/A')):7s} │ {str(agg_2spk.get('mean_sisnr_db','N/A')):7s} │ {str(agg_3spk.get('mean_sisnr_db','N/A')):7s} │
│ DER (%) ↓          │ {str(round(agg_all.get('mean_DER',0)*100,1) if agg_all.get('mean_DER') else 'N/A'):7s} │ {str(round(agg_2spk.get('mean_DER',0)*100,1) if agg_2spk.get('mean_DER') else 'N/A'):7s} │ {str(round(agg_3spk.get('mean_DER',0)*100,1) if agg_3spk.get('mean_DER') else 'N/A'):7s} │
│ Spk count acc. (%) │ {str(round(agg_all.get('speaker_count_accuracy',0)*100,1) if agg_all.get('speaker_count_accuracy') else 'N/A'):7s} │ {str(round(agg_2spk.get('speaker_count_accuracy',0)*100,1) if agg_2spk.get('speaker_count_accuracy') else 'N/A'):7s} │ {str(round(agg_3spk.get('speaker_count_accuracy',0)*100,1) if agg_3spk.get('speaker_count_accuracy') else 'N/A'):7s} │
│ False Alarm (%) ↓  │ {str(round(agg_all.get('mean_FA',0)*100,1) if agg_all.get('mean_FA') else 'N/A'):7s} │ {str(round(agg_2spk.get('mean_FA',0)*100,1) if agg_2spk.get('mean_FA') else 'N/A'):7s} │ {str(round(agg_3spk.get('mean_FA',0)*100,1) if agg_3spk.get('mean_FA') else 'N/A'):7s} │
│ Missed Speech (%) ↓│ {str(round(agg_all.get('mean_MISS',0)*100,1) if agg_all.get('mean_MISS') else 'N/A'):7s} │ {str(round(agg_2spk.get('mean_MISS',0)*100,1) if agg_2spk.get('mean_MISS') else 'N/A'):7s} │ {str(round(agg_3spk.get('mean_MISS',0)*100,1) if agg_3spk.get('mean_MISS') else 'N/A'):7s} │
└─────────────────────┴─────────┴─────────┴─────────┘
""")

if errors:
    print(f"Errors on {len(errors)} samples: {errors[:3]}")

print(f"Full data saved to: {OUTPUT_PATH}")
print("=" * 65)
