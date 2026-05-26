"""
DELBERT classification training with three selectable strategies (TRAIN_MODE).

  MODE_A_FROM_SCRATCH
    Random-init full DELBERT encoder + classifier head, trained jointly on
    labeled data. No pretraining. Matches the paper's "from-scratch" baseline.

  MODE_B_FROM_HF
    Loads a published finetuned classifier from HuggingFace
    (e.g., wanglab/delbert-wdr91), then continues full finetuning on the user's
    labeled data. Useful when the user's task is similar to one of the four
    published targets (WDR91 / LRRK2 / SETDB1 / DCAF7).

  MODE_C_PRETRAIN_FINETUNE
    Two-stage:
      Stage 1: MLM-pretrain a fresh DELBERT encoder on PRETRAIN_PARQUET
               (labels ignored — unsupervised).
      Stage 2: Build a classifier model, copy encoder weights from Stage 1,
               apply LoRA adapters to the encoder, and train the LoRA params
               + classifier head on TRAIN_PARQUET.
    Matches the paper's main recipe.

  MODE_D_HF_THEN_MLM_FINETUNE
    Three-step "domain-adaptive pretraining":
      Step 0: Load encoder weights from a published HF checkpoint (HF_MODEL_ID),
              uses HF tokenizer (count format).
      Step 1: Continue MLM training on PRETRAIN_PARQUET, starting from the
              HF-pretrained encoder rather than random init.
      Step 2: Build a classifier model, copy encoder weights from Step 1,
              apply LoRA adapters, and train on TRAIN_PARQUET.
    Useful when your unlabeled corpus is similar to but not identical to what
    the published model was pretrained on. Recommended: set MLM_LEARNING_RATE
    lower (e.g. 5e-5) and MLM_NUM_EPOCHS smaller (e.g. 10-20) for adaptation.

All modes share the same evaluation: best-val checkpoint is selected on
val_pr_auc, and the final test set is scored once at the end.

Required parquet schema (for all modes that read parquet):
  - ECFP4, FCFP4 (or FCFP6), ATOMPAIR, TOPTOR  — fingerprint columns. Each cell
    can be a length-2048 dense array OR a comma-separated string.
  - A binary label column (LABEL / label / ENRICHED / target / active / y).
    Required for classification training & test eval. Ignored during MLM
    Stage 1 of MODE_C_PRETRAIN_FINETUNE.

Run:
    python train_largedata_shay.py
"""

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from delbert.data.tokenizer import MolecularTokenizer, create_molecular_tokenizer
from delbert.data.transforms import (
    MolecularCollator,
    build_binary_vocabulary,
    molecule_to_tokens,
)
from delbert.models.delbert_model import (
    DELBERTConfig,
    DELBERTForMLM,
    DELBERTForSequenceClassification,
)


# ===========================================================================
# MODE SWITCH — set this to choose which training strategy to run
# ===========================================================================
MODE_A_FROM_SCRATCH = "from_scratch"
MODE_B_FROM_HF = "from_hf"
MODE_C_PRETRAIN_FINETUNE = "pretrain_finetune"
MODE_D_HF_THEN_MLM_FINETUNE = "hf_then_mlm_finetune"

TRAIN_MODE = MODE_A_FROM_SCRATCH
# ===========================================================================


# ---------------------------------------------------------------------------
# Run identity — controls where ALL outputs go
# Set RUN_NAME to a descriptive label (e.g. "proteinX", "wdr91_modeA_full").
# All artifacts (checkpoints, predictions, config snapshot, log) are written
# to runs/<RUN_NAME>/.  Reusing the same RUN_NAME overwrites prior outputs.
# ---------------------------------------------------------------------------
RUN_NAME = "default"
RUNS_DIR = "runs"

# ---------------------------------------------------------------------------
# Input data paths
# ---------------------------------------------------------------------------
TRAIN_PARQUET = "data/train.parquet"
TEST_PARQUET = "data/test.parquet"
PRETRAIN_PARQUET = "data/train.parquet"  # MODE_C only — set to a different file
                                         # if you have a separate unlabeled corpus

# HuggingFace model ID for MODE_B_FROM_HF
HF_MODEL_ID = "wanglab/delbert-wdr91"  # or wanglab/delbert-lrrk2, etc.

# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------
# FP_TYPES = ["ECFP4", "FCFP4", "ATOMPAIR", "TOPTOR"]
FP_TYPES = ["ECFP4"]
NBITS = 2048
LABEL_CANDIDATES = ["LABEL", "label", "ENRICHED", "enriched", "target", "active", "y"]

# Token format for fingerprint tokenization (MODE_A and MODE_C; MODE_B forces count).
#   "binary": vocab is deterministic 4*nbits + 5 specials. Each active bit becomes
#             a single token "{FP}_{bit}"; counts are ignored. Fast, no data scan.
#   "count":  vocab is built by scanning the corpus and keeping all observed
#             (fp_type, bit, count) triples that appear >= COUNT_MIN_TOKEN_FREQUENCY
#             times. Each active bit becomes "{FP}_{bit}_{count}". Matches the
#             paper and the published HF checkpoints; ~5-30K extra tokens.
TOKEN_FORMAT = "count"
COUNT_MIN_TOKEN_FREQUENCY = 1

# ---------------------------------------------------------------------------
# Architecture (paper defaults: ~70M params)
# Source: configs/experiment/pretrain_example.yaml
# Used only for MODE_A_FROM_SCRATCH and MODE_C_PRETRAIN_FINETUNE.
# MODE_B_FROM_HF reads architecture from the downloaded HF config.json.
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 640
NUM_HIDDEN_LAYERS = 16
NUM_ATTENTION_HEADS = 16
INTERMEDIATE_SIZE = 1152
MAX_POSITION_EMBEDDINGS = 1024
GLOBAL_ROPE_THETA = 160000.0
LOCAL_ATTENTION = 128
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_DROPOUT_PROB = 0.1
USE_SEGMENT_EMBEDDINGS = True
# Position-encoding mode within segments. Paper sets this to "none" so the
# model relies on segment embeddings (not absolute position) to distinguish
# fingerprint types. Default in DELBERTConfig is "absolute".
SEGMENT_POSITION_ENCODING = "none"

# ---------------------------------------------------------------------------
# Classification head (paper defaults)
# Source: configs/experiment/classify_example.yaml
# ---------------------------------------------------------------------------
NUM_LABELS = 2
CLASSIFIER_DROPOUT = 0.1
CLASSIFIER_POOLING = "mean"
POS_CLASS_WEIGHT = 1.0  # train data is balanced; raise (e.g. 15.0) for imbalanced train sets

# ---------------------------------------------------------------------------
# LoRA (used only in MODE_C_PRETRAIN_FINETUNE Stage 2)
# Source: configs/experiment/classify_example.yaml
# Requires: pip install peft
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["Wqkv"]

# ---------------------------------------------------------------------------
# Training (paper defaults — apply to classifier finetuning in all modes)
# ---------------------------------------------------------------------------
SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 50
# Inference-only batch (used for final test-set eval). Can be much larger than
# BATCH_SIZE because no gradients / optimizer state are stored. Speeds up the
# 460K-row test sweep significantly. Lower this if you OOM on a smaller GPU.
TEST_BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_CLIP_VAL = 1.0
NUM_WORKERS = 4

# Validation / early stopping (classifier training)
VAL_FRACTION = 0.1
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 0.001
MONITOR_METRIC = "pr_auc"  # 'pr_auc' or 'roc_auc'

# MLM Stage 1 (MODE_C only)
MLM_NUM_EPOCHS = 100
MLM_BATCH_SIZE = 50
MLM_LEARNING_RATE = 5e-4
MLM_PROBABILITY = 0.15
MLM_VAL_FRACTION = 0.05

# Resume: skip MLM (Stage 1) entirely and load the encoder from this checkpoint.
# Used to recover from crashes after Stage 1 completed, or to reuse one
# MLM-pretrained encoder across multiple Stage 2 experiments. Applies to
# MODE_C and MODE_D. Empty string = run Stage 1 normally.
# Example:  MLM_CHECKPOINT_TO_LOAD = "runs/default/mlm_pretrained.pt"
MLM_CHECKPOINT_TO_LOAD = ""

# Span-shuffle augmentation — randomly permutes the order of fingerprint-type
# spans within a sample (segment_ids stay attached to their tokens). Forces
# the encoder to rely on segment embeddings rather than absolute position.
# Applied in both classifier and MLM training when > 0.
# Paper uses 0.3 during MLM pretraining only.  Default 0.0 = disabled.
SPAN_SHUFFLE_PROBABILITY = 0.0

# Mixed precision (GPU only — paper uses bf16-mixed)
USE_BF16_ON_GPU = True

# Top-K used for area / weighted ranking metrics in results.csv
AREA_HITS_K = 500


# ===========================================================================
# Run setup — output folder, config snapshot, log capture
# ===========================================================================

def get_run_dir() -> str:
    """Return runs/<RUN_NAME> path (does not create it)."""
    return os.path.join(RUNS_DIR, RUN_NAME)


class _Tee:
    """Duplicate writes to two streams (e.g. stdout AND a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _collect_settings() -> dict:
    """Snapshot of every hard-coded constant — saved to config.json."""
    return {
        "run_name": RUN_NAME,
        "runs_dir": RUNS_DIR,
        "train_mode": TRAIN_MODE,
        "paths": {
            "train_parquet": TRAIN_PARQUET,
            "test_parquet": TEST_PARQUET,
            "pretrain_parquet": PRETRAIN_PARQUET,
        },
        "hf_model_id": HF_MODEL_ID,
        "fingerprints": {
            "fp_types": FP_TYPES,
            "nbits": NBITS,
            "token_format": TOKEN_FORMAT,
            "count_min_token_frequency": COUNT_MIN_TOKEN_FREQUENCY,
        },
        "label_candidates": LABEL_CANDIDATES,
        "architecture": {
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "global_rope_theta": GLOBAL_ROPE_THETA,
            "local_attention": LOCAL_ATTENTION,
            "hidden_dropout_prob": HIDDEN_DROPOUT_PROB,
            "attention_probs_dropout_prob": ATTENTION_DROPOUT_PROB,
            "use_segment_embeddings": USE_SEGMENT_EMBEDDINGS,
            "segment_position_encoding": SEGMENT_POSITION_ENCODING,
        },
        "classification_head": {
            "num_labels": NUM_LABELS,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "classifier_pooling": CLASSIFIER_POOLING,
            "pos_class_weight": POS_CLASS_WEIGHT,
        },
        "lora": {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
        },
        "training": {
            "seed": SEED,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "test_batch_size": TEST_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "gradient_clip_val": GRADIENT_CLIP_VAL,
            "num_workers": NUM_WORKERS,
        },
        "validation_early_stopping": {
            "val_fraction": VAL_FRACTION,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "monitor_metric": MONITOR_METRIC,
        },
        "mlm_stage1": {
            "num_epochs": MLM_NUM_EPOCHS,
            "batch_size": MLM_BATCH_SIZE,
            "learning_rate": MLM_LEARNING_RATE,
            "mlm_probability": MLM_PROBABILITY,
            "mlm_val_fraction": MLM_VAL_FRACTION,
            "checkpoint_to_load": MLM_CHECKPOINT_TO_LOAD,
        },
        "augmentation": {
            "span_shuffle_probability": SPAN_SHUFFLE_PROBABILITY,
        },
        "mixed_precision": {"use_bf16_on_gpu": USE_BF16_ON_GPU},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def setup_run() -> str:
    """
    Create runs/<RUN_NAME>/, write config.json, and tee stdout/stderr to
    training_log.txt inside it. Returns the run directory path.

    Re-running with the same RUN_NAME overwrites prior contents.
    """
    run_dir = get_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # Save settings snapshot
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(_collect_settings(), f, indent=2)

    # Tee stdout + stderr to log file (line-buffered)
    log_path = os.path.join(run_dir, "training_log.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Run directory: {run_dir}")
    print(f"Config snapshot: {config_path}")
    print(f"Log file: {log_path}")
    return run_dir


# ===========================================================================
# Data utilities
# ===========================================================================

def detect_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"No label column found. Tried {LABEL_CANDIDATES}. Got: {list(df.columns)}"
    )


def dense_to_sparse(arr):
    if isinstance(arr, str):
        arr = [int(x) for x in arr.split(",") if x]
    arr = np.asarray(arr, dtype=np.int32)
    nz = np.nonzero(arr)[0]
    return nz.tolist(), arr[nz].tolist()


class MolDataset(Dataset):
    """
    Pre-tokenizes all molecules in __init__.

    If label_col is None, no labels are attached to samples (used by MLM Stage 1).
    """

    def __init__(self, df, tokenizer, fp_types, label_col, max_length, token_format="binary"):
        labels = df[label_col].astype(int).tolist() if label_col is not None else None
        self.samples = []
        for i in range(len(df)):
            row = df.iloc[i]
            sparse_row = {}
            for fp in fp_types:
                idx, val = dense_to_sparse(row[fp])
                sparse_row[f"{fp}_indices"] = idx
                sparse_row[f"{fp}_values"] = val

            tokens, seg_ids = molecule_to_tokens(
                sparse_row,
                fp_types,
                return_segment_ids=True,
                token_format=token_format,
                nbits=NBITS,
            )
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                seg_ids = seg_ids[:max_length]

            sample = {"input_ids": input_ids, "segment_ids": seg_ids}
            if labels is not None:
                sample["labels"] = labels[i]
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ===========================================================================
# Optimizer / scheduler — matches ClassificationModel.configure_optimizers
# ===========================================================================

def make_optimizer(model, learning_rate, weight_decay):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if any(nd in n for nd in no_decay) else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return AdamW(groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)


# ===========================================================================
# Classifier evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_classifier(model, loader, device, autocast_dtype):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        segment_ids = batch["segment_ids"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
            )
        probs = F.softmax(outputs["logits"].float(), dim=-1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"].numpy().tolist())

    metrics = {"accuracy": accuracy_score(all_labels, all_preds)}
    if len(set(all_labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
        metrics["pr_auc"] = average_precision_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics, all_probs, all_preds, all_labels


# ===========================================================================
# Diagnostic plots — hit / enrichment / p-value curves
# ===========================================================================
# Adapted from a parallel LightGBM/baseline pipeline. Inputs are aligned
# arrays (labels, probs); we sort internally and write PNGs to run_dir.

# X-axis caps for top-of-ranking inspection.
PLOT_X_MAX = 1000               # hit curve & p-value/enrichment x-axis
PLOT_HIT_Y_MAX = 100            # hit curve y-axis cap
PLOT_N_RANDOM_RUNS = 10         # simulated random permutations on hit curve


def _sort_by_score(labels, probs):
    """Return labels sorted by probs descending (as int8 numpy array)."""
    labels = np.asarray(labels).astype(np.int8)
    probs = np.asarray(probs)
    order = np.argsort(probs)[::-1]
    return labels[order]


def plot_hit_curve(labels, probs, out_path, title=""):
    """Cumulative-hits curve with theoretical + simulated random baselines."""
    import matplotlib.pyplot as plt
    import textwrap

    y_sorted = _sort_by_score(labels, probs)
    n_total = len(y_sorted)
    n_pos = int(y_sorted.sum())
    if n_pos == 0 or n_total == 0:
        print(f"[hit_curve] No positives or empty data; skipping {out_path}.")
        return

    ranks = np.arange(1, n_total + 1)
    cum_hits = np.cumsum(y_sorted)
    random_curve = ranks * (n_pos / n_total)

    plt.figure(figsize=(8, 6))
    if PLOT_N_RANDOM_RUNS > 0:
        rng = np.random.default_rng(42)
        sim_labels = np.zeros(n_total, dtype=np.int8)
        sim_labels[:n_pos] = 1
        for i in range(PLOT_N_RANDOM_RUNS):
            sim_curve = np.cumsum(rng.permutation(sim_labels))
            kw = dict(color="lightgray", linewidth=0.8, alpha=0.3)
            if i == 0:
                kw["label"] = f"Random simulated ({PLOT_N_RANDOM_RUNS} runs)"
            plt.plot(ranks, sim_curve, **kw)
    plt.plot(ranks, random_curve, color="gray", linestyle="--", linewidth=1.5,
             label="Random expected")
    plt.plot(ranks, cum_hits, color="darkblue", linewidth=2, label="Model")

    plt.xlabel("K  (top-K predictions)")
    plt.ylabel("Hit@K  (cumulative true positives)")
    plt.title(textwrap.fill(f"Hit curve - {title}", width=60), fontsize=10)
    plt.xlim(0, min(PLOT_X_MAX, n_total))
    plt.ylim(0, PLOT_HIT_Y_MAX)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_enrichment_curve(labels, probs, out_path, title=""):
    """Fold-enrichment vs top-n: (k_hits / n_rank) / (K_pos / N_total)."""
    import matplotlib.pyplot as plt
    import textwrap

    y_sorted = _sort_by_score(labels, probs)
    n_total = len(y_sorted)
    K_pos = int(y_sorted.sum())
    if K_pos == 0 or n_total == 0:
        print(f"[enrichment_curve] No positives or empty data; skipping {out_path}.")
        return

    n_rank = np.arange(1, n_total + 1)
    k_hits = np.cumsum(y_sorted)
    enrichment = (k_hits / n_rank) / (K_pos / n_total)

    x_max = min(PLOT_X_MAX, n_total)
    enr_vis = enrichment[:x_max]

    plt.figure(figsize=(8, 6))
    plt.plot(n_rank, enrichment, color="darkgreen", linewidth=2, label="Enrichment")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="No enrichment (=1)")
    plt.xlabel("Top-n")
    plt.ylabel("Enrichment  (k/n) / (K/N)")
    plt.title(textwrap.fill(f"Enrichment vs Top-n - {title}", width=60), fontsize=10)
    plt.xlim(0, x_max)
    e_min = min(float(np.nanmin(enr_vis)), 1.0)
    e_max = max(float(np.nanmax(enr_vis)), 1.0)
    pad = max((e_max - e_min) * 0.10, 0.05)
    plt.ylim(max(0.0, e_min - pad), e_max + pad)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pvalue_curve(labels, probs, out_path, title=""):
    """Hypergeometric -log10(p-value) vs top-n. Higher = more significant."""
    import matplotlib.pyplot as plt
    import textwrap
    from scipy.stats import hypergeom

    y_sorted = _sort_by_score(labels, probs)
    N_total = len(y_sorted)
    K_pos = int(y_sorted.sum())
    if K_pos == 0 or N_total == 0:
        print(f"[pvalue_curve] No positives or empty data; skipping {out_path}.")
        return

    n_rank = np.arange(1, N_total + 1)
    k_hits = np.cumsum(y_sorted)
    p_values = hypergeom.sf(k_hits - 1, N_total, K_pos, n_rank)
    p_values = np.clip(p_values, 1e-300, 1.0)
    neg_log_p = -np.log10(p_values)

    x_max = min(PLOT_X_MAX, N_total)
    neg_log_p_vis = neg_log_p[:x_max]

    plt.figure(figsize=(8, 6))
    plt.plot(n_rank, neg_log_p, color="darkblue", linewidth=2, label="-log10(p-value)")
    plt.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1,
                label="p = 0.05  (-log10 = 1.30)")
    plt.xlabel("Top-n")
    plt.ylabel("-log10(p-value)   (higher = more significant)")
    plt.title(textwrap.fill(f"p-value vs Top-n - {title}", width=60), fontsize=10)
    plt.xlim(0, x_max)
    ymin = min(float(np.nanmin(neg_log_p_vis)), -float(np.log10(0.05)))
    ymax = max(float(np.nanmax(neg_log_p_vis)), -float(np.log10(0.05)))
    pad = max((ymax - ymin) * 0.05, 0.5)
    plt.ylim(ymin - pad, ymax + pad)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def make_diagnostic_plots(labels, probs, run_dir, title=""):
    """Generate hit, enrichment, and p-value plots into run_dir."""
    plot_hit_curve(labels, probs, os.path.join(run_dir, "hit_curve.png"), title=title)
    plot_enrichment_curve(labels, probs, os.path.join(run_dir, "enrichment_curve.png"), title=title)
    plot_pvalue_curve(labels, probs, os.path.join(run_dir, "pvalue_curve.png"), title=title)


# ===========================================================================
# Test metrics table — values written to results.csv
# Functions lifted from a parallel LightGBM/baseline pipeline so values are
# directly comparable across model types.
# ===========================================================================

def hits_and_precision_at_k(y_true, y_pred, y_scores, k):
    """Count hits in top-k by score where (true=1 AND pred=1). Returns (hits, hits/k)."""
    k = min(k, len(y_scores))
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    top_k_true = np.asarray(y_true)[top_k_idx]
    top_k_pred = np.asarray(y_pred)[top_k_idx]
    hits = int(np.sum((top_k_true == 1) & (top_k_pred == 1)))
    precision_at_k = hits / k if k > 0 else 0.0
    return hits, precision_at_k


def plate_ppv(y, y_pred, top_n: int = 128):
    """PPV among top_n score-ranked samples whose pred>0.5; mimics a screening plate."""
    y = np.atleast_1d(np.asarray(y))
    y_pred = np.atleast_1d(np.asarray(y_pred))
    stacked = np.vstack((y, y_pred)).T[y_pred.argsort()[::-1]][:top_n, :]
    stacked = stacked[stacked[:, 1] > 0.5]
    if len(stacked) == 0:
        return 0.0
    return float(np.sum(stacked[:, 0]) / len(stacked))


def _ideal_area_hits_at_k(P, K):
    P = int(P); K = int(K)
    if P >= K:
        return K * (K + 1) // 2
    return P * (P + 1) // 2 + P * (K - P)


def area_hits_at_k(y_true, y_score, K):
    """Σ_{k=1..K} hits@k. Earlier hits weighted more (counted at every k from rank to K)."""
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0
    K = min(K, len(y_true))
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order][:K].astype(int)
    return int(np.cumsum(y_sorted).sum())


def area_hits_at_k_norm(y_true, y_score, K):
    """area_hits_at_k normalized to [0, 1] by the perfect-ranker maximum."""
    P = int(np.sum(np.asarray(y_true) == 1))
    ideal = _ideal_area_hits_at_k(P, K)
    if ideal == 0:
        return 0.0
    return float(area_hits_at_k(y_true, y_score, K) / ideal)


def log_weighted_hits_at_k(y_true, y_score, K):
    """Σ_{k=1..K} cum_hits(k) / log2(k+1). Front-loads early ranks vs. plain area."""
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0.0
    K = min(K, len(y_true))
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order][:K].astype(int)
    cum = np.cumsum(y_sorted)
    weights = 1.0 / np.log2(np.arange(1, K + 1) + 1)
    return float(np.sum(cum * weights))


def ndcg_at_k(y_true, y_score, K):
    """Standard NDCG@K with binary relevance, in [0, 1]."""
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0.0
    K = min(K, len(y_true))
    y_arr = np.asarray(y_true).astype(int)
    P = int(np.sum(y_arr == 1))
    if P == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    y_sorted = y_arr[order][:K]
    discounts = 1.0 / np.log2(np.arange(1, K + 1) + 1)
    dcg = float(np.sum(y_sorted * discounts))
    ideal_K = min(P, K)
    ideal_dcg = float(np.sum(discounts[:ideal_K]))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def bedroc_at_k(y_true, y_score, K, alpha=20.0):
    """BEDROC-style score restricted to top-K, in [0, 1]. Aggressive front-loading."""
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0.0
    K = min(K, len(y_true))
    order = np.argsort(y_score)[::-1][:K]
    y_top = np.asarray(y_true)[order].astype(int)
    P = int(np.sum(y_top == 1))
    if P == 0:
        return 0.0
    hit_ranks = np.where(y_top == 1)[0] + 1
    score = float(np.sum(np.exp(-alpha * hit_ranks / K)))
    ideal = float(np.sum(np.exp(-alpha * np.arange(1, P + 1) / K)))
    if ideal == 0:
        return 0.0
    return score / ideal


def compute_test_metrics_table(labels, preds, probs, K=AREA_HITS_K):
    """Compute all metrics requested for results.csv. Returns dict with Test_ prefix."""
    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    y_proba = np.asarray(probs)

    hits_50, _ = hits_and_precision_at_k(y_true, y_pred, y_proba, 50)
    hits_100, _ = hits_and_precision_at_k(y_true, y_pred, y_proba, 100)
    hits_200, prec_200 = hits_and_precision_at_k(y_true, y_pred, y_proba, 200)
    hits_500, prec_500 = hits_and_precision_at_k(y_true, y_pred, y_proba, 500)
    total_hits = int(np.sum((y_true == 1) & (y_pred == 1)))

    metrics = {
        "Test_Accuracy": accuracy_score(y_true, y_pred),
        "Test_Precision": precision_score(y_true, y_pred, zero_division=0),
        "Test_Recall": recall_score(y_true, y_pred, zero_division=0),
        "Test_F1Score": f1_score(y_true, y_pred, zero_division=0),
        "Test_PlatePPV": plate_ppv(y_true, y_pred, top_n=128),
        "Test_HitsAt50": hits_50,
        "Test_HitsAt100": hits_100,
        "Test_HitsAt200": hits_200,
        "Test_HitsAt500": hits_500,
        "Test_PrecisionAt200": prec_200,
        "Test_PrecisionAt500": prec_500,
        "Test_TotalHits": total_hits,
        f"Test_AreaHitsAt{K}": area_hits_at_k(y_true, y_proba, K),
        f"Test_AreaHitsAt{K}_norm": area_hits_at_k_norm(y_true, y_proba, K),
        f"Test_LogWeightedHitsAt{K}": log_weighted_hits_at_k(y_true, y_proba, K),
        f"Test_NDCG_at_{K}": ndcg_at_k(y_true, y_proba, K),
        f"Test_BEDROC_alpha20_at{K}": bedroc_at_k(y_true, y_proba, K, alpha=20.0),
    }
    return metrics


def save_results_csv(metrics_dict, run_dir):
    """Write a single-row results.csv with the metrics from compute_test_metrics_table."""
    out_path = os.path.join(run_dir, "results.csv")
    pd.DataFrame([metrics_dict]).to_csv(out_path, index=False)
    print(f"Saved test metrics table to {out_path}")
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


# ===========================================================================
# Tokenizer / model builders
# ===========================================================================

def build_binary_tokenizer():
    """Deterministic binary tokenizer used when TOKEN_FORMAT == 'binary'."""
    vocab_data = build_binary_vocabulary(FP_TYPES, nbits=NBITS)
    tokenizer = create_molecular_tokenizer(
        vocabulary=vocab_data["token_to_id"],
        fingerprint_types=FP_TYPES,
        token_format="binary",
        fingerprint_nbits=NBITS,
    )
    return tokenizer


def build_count_vocabulary_from_dataframe(df, fp_types, nbits, min_frequency=1):
    """Scan a pandas DataFrame to build a count-mode vocabulary.

    Counts every (fp_type, bit, count) triple that appears, keeps the ones with
    frequency >= min_frequency, sorts deterministically, and adds the 5 special
    tokens. Returns a dict in the format expected by create_molecular_tokenizer.
    """
    from collections import Counter
    counter = Counter()
    for i in range(len(df)):
        row = df.iloc[i]
        for fp in fp_types:
            indices, values = dense_to_sparse(row[fp])
            for idx, count in zip(indices, values):
                if count > 0 and idx < nbits:
                    counter[f"{fp}_{idx}_{count}"] += 1

    filtered = {tok: c for tok, c in counter.items() if c >= min_frequency}
    vocab_tokens = sorted(filtered.keys())
    token_to_id = {tok: i for i, tok in enumerate(vocab_tokens)}
    for special in ("<unk>", "<pad>", "<mask>", "<cls>", "<sep>"):
        if special not in token_to_id:
            token_to_id[special] = len(token_to_id)

    return {
        "token_to_id": token_to_id,
        "id_to_token": {v: k for k, v in token_to_id.items()},
        "vocab_size": len(token_to_id),
        "token_counts": filtered,
    }


def build_count_tokenizer(df, min_frequency=None):
    """Count-mode tokenizer built from `df` (typically the train/pretrain corpus)."""
    if min_frequency is None:
        min_frequency = COUNT_MIN_TOKEN_FREQUENCY
    print(
        f"Building count vocabulary from {len(df):,} molecules "
        f"(min_frequency={min_frequency})..."
    )
    vocab_data = build_count_vocabulary_from_dataframe(
        df, FP_TYPES, NBITS, min_frequency=min_frequency
    )
    print(f"Count vocab built: {vocab_data['vocab_size']:,} tokens "
          f"({len(vocab_data['token_counts']):,} non-special).")
    tokenizer = create_molecular_tokenizer(
        vocabulary=vocab_data["token_to_id"],
        fingerprint_types=FP_TYPES,
        token_format="count",
        fingerprint_nbits=NBITS,
    )
    return tokenizer


def build_tokenizer_for_mode(df_for_count_vocab=None):
    """Dispatcher honoring TOKEN_FORMAT. For 'count', df_for_count_vocab is required."""
    if TOKEN_FORMAT == "binary":
        return build_binary_tokenizer()
    if TOKEN_FORMAT == "count":
        if df_for_count_vocab is None:
            raise ValueError(
                "TOKEN_FORMAT='count' requires a DataFrame to scan for vocab."
            )
        return build_count_tokenizer(df_for_count_vocab)
    raise ValueError(
        f"Unknown TOKEN_FORMAT={TOKEN_FORMAT!r}. Choose 'binary' or 'count'."
    )


def build_classifier_from_scratch(vocab_size: int) -> DELBERTForSequenceClassification:
    config = DELBERTConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        global_rope_theta=GLOBAL_ROPE_THETA,
        local_attention=LOCAL_ATTENTION,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=ATTENTION_DROPOUT_PROB,
        classifier_dropout=CLASSIFIER_DROPOUT,
        classifier_pooling=CLASSIFIER_POOLING,
        use_segment_embeddings=USE_SEGMENT_EMBEDDINGS,
        segment_position_encoding=SEGMENT_POSITION_ENCODING,
    )
    return DELBERTForSequenceClassification(config, num_labels=NUM_LABELS)


def build_mlm_from_scratch(vocab_size: int) -> DELBERTForMLM:
    config = DELBERTConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        global_rope_theta=GLOBAL_ROPE_THETA,
        local_attention=LOCAL_ATTENTION,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=ATTENTION_DROPOUT_PROB,
        use_segment_embeddings=USE_SEGMENT_EMBEDDINGS,
        segment_position_encoding=SEGMENT_POSITION_ENCODING,
    )
    return DELBERTForMLM(config)


def load_hf_classifier(repo_id: str):
    """Download a published DELBERT classifier from HF and instantiate it.

    Mirrors the pattern in inference/predict.py:load_model().
    Returns (model, tokenizer, token_format).
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print(f"Downloading model from HuggingFace: {repo_id}")
    local_dir = Path(snapshot_download(repo_id=repo_id))

    with open(local_dir / "config.json") as f:
        config_dict = json.load(f)
    num_labels = config_dict.pop("num_labels", 2)
    for k in ("architectures", "model_type", "lora_merged"):
        config_dict.pop(k, None)

    config = DELBERTConfig(**config_dict)
    model = DELBERTForSequenceClassification(config, num_labels=num_labels)

    safetensors_path = local_dir / "model.safetensors"
    pytorch_path = local_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(str(pytorch_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights in {local_dir}")
    model.load_state_dict(state_dict)

    tokenizer = MolecularTokenizer.from_pretrained(str(local_dir))
    # Published checkpoints use count-format tokens (see predict.py).
    return model, tokenizer, "count"


def copy_encoder_weights(src_mlm: DELBERTForMLM, dst_cls: DELBERTForSequenceClassification):
    """Copy encoder + segment embedding weights from MLM model into classifier model."""
    dst_cls.encoder.load_state_dict(src_mlm.encoder.state_dict())
    if (
        getattr(dst_cls, "segment_embedding_layer", None) is not None
        and getattr(src_mlm, "segment_embedding_layer", None) is not None
    ):
        dst_cls.segment_embedding_layer.load_state_dict(
            src_mlm.segment_embedding_layer.state_dict()
        )


def build_mlm_from_hf_classifier(hf_classifier: DELBERTForSequenceClassification) -> DELBERTForMLM:
    """Build a DELBERTForMLM with encoder weights copied from an HF-loaded classifier.

    The MLM head is freshly initialized (the classifier didn't have one). The
    config is reused so vocab size, hidden size, segment settings all match.
    """
    mlm_model = DELBERTForMLM(hf_classifier.config)
    mlm_model.encoder.load_state_dict(hf_classifier.encoder.state_dict())
    src_seg = getattr(hf_classifier, "segment_embedding_layer", None)
    dst_seg = getattr(mlm_model, "segment_embedding_layer", None)
    if src_seg is not None and dst_seg is not None:
        dst_seg.load_state_dict(src_seg.state_dict())
    return mlm_model


def load_mlm_from_checkpoint(ckpt_path: str, device) -> DELBERTForMLM:
    """Load a DELBERTForMLM previously saved by mlm_pretrain_loop.

    The saved file contains {'model_state_dict': ..., 'config': dict}. We pop
    a few non-config fields that may have been added by transformers' config
    serializer, then reconstruct DELBERTConfig and the model.
    """
    print(f"Loading MLM checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config_dict = dict(ckpt["config"])
    for key in ("architectures", "model_type", "transformers_version", "lora_merged"):
        config_dict.pop(key, None)
    config = DELBERTConfig(**config_dict)
    model = DELBERTForMLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


# ===========================================================================
# Training loops
# ===========================================================================

def train_classifier_loop(
    model,
    train_loader,
    val_loader,
    test_loader_builder,
    device,
    autocast_dtype,
    config_dict,
    run_name="run",
):
    """Standard classification training loop with cosine LR, early stopping, best-ckpt."""
    class_weights = torch.tensor([1.0, POS_CLASS_WEIGHT], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Loss: weighted CE  weights=[1.0, {POS_CLASS_WEIGHT}]")

    optim = make_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    print(f"Total steps: {total_steps},  warmup steps: {warmup_steps}")

    run_dir = get_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    best_ckpt_path = os.path.join(run_dir, f"{run_name}_best.pt")

    print(f"\n--- Classifier training ({NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}) ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            segment_ids = batch["segment_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
                loss = loss_fn(outputs["logits"].float(), labels)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                GRADIENT_CLIP_VAL,
            )
            optim.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_metrics, *_ = evaluate_classifier(model, val_loader, device, autocast_dtype)
        current = val_metrics[MONITOR_METRIC]

        improved = current > best_metric + EARLY_STOP_MIN_DELTA
        if improved:
            best_metric = current
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config_dict,
                "val_metrics": val_metrics,
                "train_mode": TRAIN_MODE,
            }, best_ckpt_path)
        else:
            epochs_without_improve += 1

        flag = " *" if improved else ""
        print(
            f"  Epoch {epoch:2d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.3f} | "
            f"val_roc_auc={val_metrics['roc_auc']:.3f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}{flag}"
        )

        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best: epoch {best_epoch}, val_{MONITOR_METRIC}={best_metric:.4f})")
            break

    # Free the optimizer + scheduler before returning — they're function-local
    # and hold significant GPU memory.
    del optim, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nTraining done. Best checkpoint: {best_ckpt_path}")
    print(f"  best_epoch={best_epoch}, val_{MONITOR_METRIC}={best_metric:.4f}")
    return best_ckpt_path


def run_final_test_eval(model, best_ckpt_path, test_loader_builder,
                       device, autocast_dtype, run_dir, title=""):
    """Load the best checkpoint, build the test loader, evaluate, save artifacts.

    Called AFTER the caller has freed all train/val resources so the test
    parquet read fits in RAM. This separation is important on small instances
    (e.g. 16 GB g2-standard-4) where the combined train + test memory
    footprint would otherwise OOM.
    """
    print(f"\n--- Loading best checkpoint ---")
    print(f"  Path: {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Building test loader (deferred read) ---")
    test_loader = test_loader_builder()

    print("\n--- Final test-set evaluation ---")
    test_metrics, probs, preds, labels = evaluate_classifier(
        model, test_loader, device, autocast_dtype,
    )
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")

    predictions_path = os.path.join(run_dir, "predictions_test.csv")
    pd.DataFrame({"label": labels, "pred": preds, "prob_active": probs}).to_csv(
        predictions_path, index=False,
    )
    print(f"\nSaved test predictions to {predictions_path}")

    print("\n--- Generating diagnostic plots ---")
    make_diagnostic_plots(labels, probs, run_dir, title=title)

    print("\n--- Computing test metrics table ---")
    metrics_table = compute_test_metrics_table(labels, preds, probs, K=AREA_HITS_K)
    save_results_csv(metrics_table, run_dir)


def mlm_pretrain_loop(model, train_loader, val_loader, device, autocast_dtype):
    """MLM pretraining loop. Returns the trained model."""
    optim = make_optimizer(model, MLM_LEARNING_RATE, WEIGHT_DECAY)
    total_steps = MLM_NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    print(f"MLM total steps: {total_steps},  warmup steps: {warmup_steps}")

    print(f"\n--- MLM pretraining ({MLM_NUM_EPOCHS} epochs, batch_size={MLM_BATCH_SIZE}, lr={MLM_LEARNING_RATE}) ---")
    for epoch in range(1, MLM_NUM_EPOCHS + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            segment_ids = batch["segment_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    labels=labels,
                )
                loss = outputs["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optim.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Quick val MLM loss
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                segment_ids = batch["segment_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                                segment_ids=segment_ids, labels=labels)
                val_loss += out["loss"].item()
                val_n += 1
        val_loss /= max(val_n, 1)

        print(f"  MLM Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # Save MLM-pretrained encoder
    run_dir = get_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    mlm_ckpt_path = os.path.join(run_dir, "mlm_pretrained.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
    }, mlm_ckpt_path)
    print(f"\nMLM checkpoint saved to {mlm_ckpt_path}")
    return model


# ===========================================================================
# Mode runners
# ===========================================================================

def _make_loaders_for_classification(train_df, tokenizer, label_col, token_format, device):
    """Stratified internal train/val split + train and val DataLoaders.

    The test DataLoader is built later by build_test_loader_from_path() — only
    once training has finished and all Stage-1 / training memory has been
    released — so the 460K-row test parquet read doesn't compete with the
    classifier's training memory footprint.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=train_df[label_col].values,
    )
    train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_df_split = train_df.iloc[val_idx].reset_index(drop=True)
    print(f"Internal split: train={len(train_df_split):,}, val={len(val_df_split):,}")

    max_len = MAX_POSITION_EMBEDDINGS
    train_ds = MolDataset(train_df_split, tokenizer, FP_TYPES, label_col, max_len, token_format)
    val_ds = MolDataset(val_df_split, tokenizer, FP_TYPES, label_col, max_len, token_format)

    seq_lens = [len(s["input_ids"]) for s in train_ds.samples]
    print(f"Train seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

    # Span shuffling is only applied during training; eval stays deterministic.
    train_collator = MolecularCollator(
        pad_token_id=tokenizer.pad_token_id,
        span_shuffle_probability=SPAN_SHUFFLE_PROBABILITY,
    )
    eval_collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    pin_memory = device.type == "cuda"
    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=pin_memory,
                     persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=train_collator, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=eval_collator, **dl_kwargs)
    return train_loader, val_loader


def build_test_loader_from_path(test_parquet_path, tokenizer, label_col, token_format, device):
    """Read the test parquet (column-filtered) and build its DataLoader.

    Called AFTER classifier training finishes — at the point where MLM
    resources are gone, training optimizer state is gone, and free RAM is at
    its peak. This avoids the OOM that happens when the test parquet is
    pre-loaded alongside training resources.

    Only reads the columns we actually need (4 FP columns + label) to keep
    the in-memory DataFrame small even for very large test files.
    """
    cols_needed = FP_TYPES + [label_col]
    print(f"Reading {test_parquet_path} (columns: {cols_needed})...")
    test_df = pd.read_parquet(test_parquet_path, columns=cols_needed)
    print(f"Test:      {len(test_df):,} from {test_parquet_path}")
    print(f"Test class balance: {test_df[label_col].value_counts().to_dict()}")

    test_ds = MolDataset(
        test_df, tokenizer, FP_TYPES, label_col,
        MAX_POSITION_EMBEDDINGS, token_format,
    )
    eval_collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    pin_memory = device.type == "cuda"
    print(f"Test DataLoader batch_size={TEST_BATCH_SIZE} (inference-only).")
    return DataLoader(
        test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
        collate_fn=eval_collator, num_workers=NUM_WORKERS,
        pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )


def run_from_scratch(device, autocast_dtype):
    """MODE_A: random-init full model, train classifier on labeled data."""
    print("\n" + "=" * 60)
    print("MODE A: From scratch (no pretraining)")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    print(f"Label column: '{label_col}'")
    print(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")

    tokenizer = build_tokenizer_for_mode(df_for_count_vocab=train_df)
    print(f"Vocab size: {tokenizer.vocab_size}  |  token_format: {TOKEN_FORMAT}")

    train_loader, val_loader = _make_loaders_for_classification(
        train_df, tokenizer, label_col, token_format=TOKEN_FORMAT, device=device
    )

    model = build_classifier_from_scratch(tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, TOKEN_FORMAT, device,
        )

    best_ckpt_path = train_classifier_loop(
        model, train_loader, val_loader, _test_loader_builder,
        device, autocast_dtype, model.config.to_dict(), run_name="mode_a",
    )

    # Free training-side memory before final test eval so the test parquet
    # read doesn't compete with persistent DataLoader workers + train_df + ds.
    del train_loader, val_loader, train_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training-side resources released.")

    run_final_test_eval(
        model, best_ckpt_path, _test_loader_builder,
        device, autocast_dtype, get_run_dir(), title=RUN_NAME,
    )


def run_from_hf(device, autocast_dtype):
    """MODE_B: load published HF classifier, continue full finetuning."""
    print("\n" + "=" * 60)
    print(f"MODE B: From HuggingFace ({HF_MODEL_ID})")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    print(f"Label column: '{label_col}'")

    model, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  |  token_format: {token_format}  |  vocab: {tokenizer.vocab_size}")

    train_loader, val_loader = _make_loaders_for_classification(
        train_df, tokenizer, label_col, token_format=token_format, device=device
    )

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, token_format, device,
        )

    best_ckpt_path = train_classifier_loop(
        model, train_loader, val_loader, _test_loader_builder,
        device, autocast_dtype, model.config.to_dict(), run_name="mode_b",
    )

    del train_loader, val_loader, train_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training-side resources released.")

    run_final_test_eval(
        model, best_ckpt_path, _test_loader_builder,
        device, autocast_dtype, get_run_dir(), title=RUN_NAME,
    )


def run_pretrain_finetune(device, autocast_dtype):
    """MODE_C: MLM pretrain (Stage 1) → LoRA classifier finetune (Stage 2)."""
    print("\n" + "=" * 60)
    print("MODE C: MLM pretrain → LoRA finetune")
    print("=" * 60)

    # ---- Stage 1: MLM pretraining (or resume from checkpoint) ----
    pretrain_df = pd.read_parquet(PRETRAIN_PARQUET)
    print(f"Pretrain corpus: {len(pretrain_df):,} molecules (labels ignored)")
    tokenizer = build_tokenizer_for_mode(df_for_count_vocab=pretrain_df)
    print(f"Vocab size: {tokenizer.vocab_size}  |  token_format: {TOKEN_FORMAT}")

    if MLM_CHECKPOINT_TO_LOAD:
        print(f"\n[Stage 1] SKIPPED — loading MLM checkpoint from {MLM_CHECKPOINT_TO_LOAD}")
        mlm_model = load_mlm_from_checkpoint(MLM_CHECKPOINT_TO_LOAD, device)
        print(f"MLM model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")
        # Only the DataFrame is in scope; release it before Stage 2.
        del pretrain_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"\n[Stage 1] MLM pretraining on {PRETRAIN_PARQUET}")

        # Internal split for MLM val loss monitoring
        pre_train_idx, pre_val_idx = train_test_split(
            np.arange(len(pretrain_df)),
            test_size=MLM_VAL_FRACTION,
            random_state=SEED,
        )
        mlm_train_df = pretrain_df.iloc[pre_train_idx].reset_index(drop=True)
        mlm_val_df = pretrain_df.iloc[pre_val_idx].reset_index(drop=True)

        mlm_train_ds = MolDataset(mlm_train_df, tokenizer, FP_TYPES, label_col=None,
                                  max_length=MAX_POSITION_EMBEDDINGS, token_format=TOKEN_FORMAT)
        mlm_val_ds = MolDataset(mlm_val_df, tokenizer, FP_TYPES, label_col=None,
                                max_length=MAX_POSITION_EMBEDDINGS, token_format=TOKEN_FORMAT)

        mlm_train_collator = MolecularCollator(
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            mlm_probability=MLM_PROBABILITY,
            vocab_size=tokenizer.vocab_size,
            span_shuffle_probability=SPAN_SHUFFLE_PROBABILITY,
        )
        mlm_eval_collator = MolecularCollator(
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            mlm_probability=MLM_PROBABILITY,
            vocab_size=tokenizer.vocab_size,
        )
        pin_memory = device.type == "cuda"
        dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=pin_memory,
                         persistent_workers=NUM_WORKERS > 0)
        mlm_train_loader = DataLoader(mlm_train_ds, batch_size=MLM_BATCH_SIZE, shuffle=True,
                                      collate_fn=mlm_train_collator, **dl_kwargs)
        mlm_val_loader = DataLoader(mlm_val_ds, batch_size=MLM_BATCH_SIZE, shuffle=False,
                                    collate_fn=mlm_eval_collator, **dl_kwargs)

        mlm_model = build_mlm_from_scratch(tokenizer.vocab_size).to(device)
        print(f"MLM model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")

        mlm_model = mlm_pretrain_loop(mlm_model, mlm_train_loader, mlm_val_loader,
                                      device, autocast_dtype)

        # Free Stage-1 resources before Stage 2 reads the (potentially huge)
        # test parquet — otherwise system RAM can OOM on 16 GB instances.
        del mlm_train_loader, mlm_val_loader, mlm_train_ds, mlm_val_ds
        del mlm_train_collator, mlm_eval_collator
        del pretrain_df, mlm_train_df, mlm_val_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Stage 1 resources released.")

    # ---- Stage 2: Classifier finetune with LoRA ----
    print(f"\n[Stage 2] Classifier finetuning on {TRAIN_PARQUET} with LoRA")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    print(f"Label column: '{label_col}'")
    print(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")

    cls_model = build_classifier_from_scratch(tokenizer.vocab_size).to(device)
    copy_encoder_weights(mlm_model, cls_model)
    print("Copied encoder + segment-embedding weights from MLM model into classifier.")

    # MLM model's job is done — its weights are now in cls_model.encoder.
    del mlm_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Apply LoRA via the repo's strategy pattern (requires `pip install peft`)
    from delbert.models.finetuning_strategies import get_finetuning_strategy
    strategy = get_finetuning_strategy(
        "lora",
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    strategy.apply(cls_model)

    total = sum(p.numel() for p in cls_model.parameters())
    trainable = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    print(f"Classifier parameters: total={total:,}  trainable={trainable:,}  ({100 * trainable / total:.2f}% via LoRA)")

    train_loader, val_loader = _make_loaders_for_classification(
        train_df, tokenizer, label_col, token_format=TOKEN_FORMAT, device=device,
    )

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, TOKEN_FORMAT, device,
        )

    best_ckpt_path = train_classifier_loop(
        cls_model, train_loader, val_loader, _test_loader_builder,
        device, autocast_dtype, cls_model.config.to_dict(), run_name="mode_c",
    )

    del train_loader, val_loader, train_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training-side resources released.")

    run_final_test_eval(
        cls_model, best_ckpt_path, _test_loader_builder,
        device, autocast_dtype, get_run_dir(), title=RUN_NAME,
    )


def run_hf_then_mlm_finetune(device, autocast_dtype):
    """MODE_D: HF weights -> continued MLM on PRETRAIN_PARQUET -> LoRA finetune."""
    print("\n" + "=" * 60)
    print(f"MODE D: HF ({HF_MODEL_ID}) -> MLM on PRETRAIN_PARQUET -> LoRA finetune")
    print("=" * 60)

    # ---- Load HF model: get encoder weights + tokenizer + token_format ----
    print(f"\nLoading HF checkpoint {HF_MODEL_ID}...")
    hf_cls_model, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)
    print(f"Tokenizer vocab: {tokenizer.vocab_size}  |  token_format: {token_format}")

    # ---- Stage 1: continued MLM on PRETRAIN_PARQUET ----
    print(f"\n[Stage 1] MLM pretraining on {PRETRAIN_PARQUET}")
    print("Note: starting from HF-pretrained encoder. Consider lowering")
    print("      MLM_LEARNING_RATE (e.g. 5e-5) and MLM_NUM_EPOCHS (e.g. 10-20).")
    pretrain_df = pd.read_parquet(PRETRAIN_PARQUET)
    print(f"Pretrain corpus: {len(pretrain_df):,} molecules (labels ignored)")

    mlm_model = build_mlm_from_hf_classifier(hf_cls_model).to(device)
    print(f"MLM model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")

    # Free HF classifier — only the encoder weights mattered.
    del hf_cls_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pre_train_idx, pre_val_idx = train_test_split(
        np.arange(len(pretrain_df)), test_size=MLM_VAL_FRACTION, random_state=SEED,
    )
    mlm_train_df = pretrain_df.iloc[pre_train_idx].reset_index(drop=True)
    mlm_val_df = pretrain_df.iloc[pre_val_idx].reset_index(drop=True)

    mlm_train_ds = MolDataset(
        mlm_train_df, tokenizer, FP_TYPES, label_col=None,
        max_length=MAX_POSITION_EMBEDDINGS, token_format=token_format,
    )
    mlm_val_ds = MolDataset(
        mlm_val_df, tokenizer, FP_TYPES, label_col=None,
        max_length=MAX_POSITION_EMBEDDINGS, token_format=token_format,
    )

    mlm_train_collator = MolecularCollator(
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        mlm_probability=MLM_PROBABILITY,
        vocab_size=tokenizer.vocab_size,
        span_shuffle_probability=SPAN_SHUFFLE_PROBABILITY,
    )
    mlm_eval_collator = MolecularCollator(
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        mlm_probability=MLM_PROBABILITY,
        vocab_size=tokenizer.vocab_size,
    )
    pin_memory = device.type == "cuda"
    dl_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=pin_memory,
                     persistent_workers=NUM_WORKERS > 0)
    mlm_train_loader = DataLoader(mlm_train_ds, batch_size=MLM_BATCH_SIZE, shuffle=True,
                                  collate_fn=mlm_train_collator, **dl_kwargs)
    mlm_val_loader = DataLoader(mlm_val_ds, batch_size=MLM_BATCH_SIZE, shuffle=False,
                                collate_fn=mlm_eval_collator, **dl_kwargs)

    mlm_model = mlm_pretrain_loop(mlm_model, mlm_train_loader, mlm_val_loader,
                                  device, autocast_dtype)

    # ---- Stage 2: LoRA classifier finetune on TRAIN_PARQUET ----
    print(f"\n[Stage 2] Classifier finetuning on {TRAIN_PARQUET} with LoRA")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    print(f"Label column: '{label_col}'")
    print(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")

    cls_model = DELBERTForSequenceClassification(mlm_model.config, num_labels=NUM_LABELS).to(device)
    copy_encoder_weights(mlm_model, cls_model)
    print("Copied encoder + segment-embedding weights from MLM model into classifier.")

    from delbert.models.finetuning_strategies import get_finetuning_strategy
    strategy = get_finetuning_strategy(
        "lora",
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    strategy.apply(cls_model)

    total = sum(p.numel() for p in cls_model.parameters())
    trainable = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    print(f"Classifier parameters: total={total:,}  trainable={trainable:,}  "
          f"({100 * trainable / total:.2f}% via LoRA)")

    train_loader, val_loader = _make_loaders_for_classification(
        train_df, tokenizer, label_col, token_format=token_format, device=device,
    )

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, token_format, device,
        )

    best_ckpt_path = train_classifier_loop(
        cls_model, train_loader, val_loader, _test_loader_builder,
        device, autocast_dtype, cls_model.config.to_dict(), run_name="mode_d",
    )

    del train_loader, val_loader, train_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training-side resources released.")

    run_final_test_eval(
        cls_model, best_ckpt_path, _test_loader_builder,
        device, autocast_dtype, get_run_dir(), title=RUN_NAME,
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    setup_run()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and USE_BF16_ON_GPU) else None
    print(f"Device: {device}  |  autocast: {autocast_dtype}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"TRAIN_MODE: {TRAIN_MODE}")

    if TRAIN_MODE == MODE_A_FROM_SCRATCH:
        run_from_scratch(device, autocast_dtype)
    elif TRAIN_MODE == MODE_B_FROM_HF:
        run_from_hf(device, autocast_dtype)
    elif TRAIN_MODE == MODE_C_PRETRAIN_FINETUNE:
        run_pretrain_finetune(device, autocast_dtype)
    elif TRAIN_MODE == MODE_D_HF_THEN_MLM_FINETUNE:
        run_hf_then_mlm_finetune(device, autocast_dtype)
    else:
        raise ValueError(
            f"Unknown TRAIN_MODE: {TRAIN_MODE!r}. Choose from: "
            f"{MODE_A_FROM_SCRATCH}, {MODE_B_FROM_HF}, "
            f"{MODE_C_PRETRAIN_FINETUNE}, {MODE_D_HF_THEN_MLM_FINETUNE}"
        )


if __name__ == "__main__":
    main()
