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

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

TRAIN_MODE = MODE_A_FROM_SCRATCH
# ===========================================================================


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PARQUET = "data/train.parquet"
TEST_PARQUET = "data/test.parquet"
PRETRAIN_PARQUET = "data/train.parquet"  # MODE_C only — set to a different file
                                         # if you have a separate unlabeled corpus
CHECKPOINT_DIR = "checkpoints"
PREDICTIONS_CSV = "predictions_test.csv"
MLM_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "mlm_pretrained.pt")

# HuggingFace model ID for MODE_B_FROM_HF
HF_MODEL_ID = "wanglab/delbert-wdr91"  # or wanglab/delbert-lrrk2, etc.

# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------
# FP_TYPES = ["ECFP4", "FCFP4", "ATOMPAIR", "TOPTOR"]
FP_TYPES = ["ECFP4"]
NBITS = 2048
LABEL_CANDIDATES = ["LABEL", "label", "ENRICHED", "enriched", "target", "active", "y"]

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
MLM_NUM_EPOCHS = 5
MLM_BATCH_SIZE = 50
MLM_LEARNING_RATE = 5e-4
MLM_PROBABILITY = 0.15
MLM_VAL_FRACTION = 0.05

# Mixed precision (GPU only — paper uses bf16-mixed)
USE_BF16_ON_GPU = True


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
# Tokenizer / model builders
# ===========================================================================

def build_binary_tokenizer():
    """Deterministic binary tokenizer used by MODE_A and MODE_C."""
    vocab_data = build_binary_vocabulary(FP_TYPES, nbits=NBITS)
    tokenizer = create_molecular_tokenizer(
        vocabulary=vocab_data["token_to_id"],
        fingerprint_types=FP_TYPES,
        token_format="binary",
        fingerprint_nbits=NBITS,
    )
    return tokenizer


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


# ===========================================================================
# Training loops
# ===========================================================================

def train_classifier_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
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

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pt")

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

    # Load best and evaluate on test
    print(f"\n--- Loading best checkpoint (epoch {best_epoch}, val_{MONITOR_METRIC}={best_metric:.4f}) ---")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n--- Final test-set evaluation ---")
    test_metrics, probs, preds, labels = evaluate_classifier(model, test_loader, device, autocast_dtype)
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")

    pd.DataFrame({"label": labels, "pred": preds, "prob_active": probs}).to_csv(
        PREDICTIONS_CSV, index=False
    )
    print(f"\nSaved test predictions to {PREDICTIONS_CSV}")
    print(f"Best checkpoint: {best_ckpt_path}")


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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
    }, MLM_CHECKPOINT_PATH)
    print(f"\nMLM checkpoint saved to {MLM_CHECKPOINT_PATH}")
    return model


# ===========================================================================
# Mode runners
# ===========================================================================

def _make_loaders_for_classification(train_df, test_df, tokenizer, label_col, token_format, device):
    """Stratified internal train/val split + DataLoaders."""
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
    test_ds = MolDataset(test_df, tokenizer, FP_TYPES, label_col, max_len, token_format)

    seq_lens = [len(s["input_ids"]) for s in train_ds.samples]
    print(f"Train seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

    collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    pin_memory = device.type == "cuda"
    common = dict(num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
                  collate_fn=collator)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def run_from_scratch(device, autocast_dtype):
    """MODE_A: random-init full model, train classifier on labeled data."""
    print("\n" + "=" * 60)
    print("MODE A: From scratch (no pretraining)")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")
    print(f"Test:      {len(test_df):,} from {TEST_PARQUET}")

    label_col = detect_label_column(train_df)
    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' missing from test parquet")
    print(f"Label column: '{label_col}'")
    print(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")
    print(f"Test class balance:      {test_df[label_col].value_counts().to_dict()}")

    tokenizer = build_binary_tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_loader, val_loader, test_loader = _make_loaders_for_classification(
        train_df, test_df, tokenizer, label_col, token_format="binary", device=device
    )

    model = build_classifier_from_scratch(tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    train_classifier_loop(
        model, train_loader, val_loader, test_loader,
        device, autocast_dtype, model.config.to_dict(), run_name="mode_a",
    )


def run_from_hf(device, autocast_dtype):
    """MODE_B: load published HF classifier, continue full finetuning."""
    print("\n" + "=" * 60)
    print(f"MODE B: From HuggingFace ({HF_MODEL_ID})")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")
    print(f"Test:      {len(test_df):,} from {TEST_PARQUET}")

    label_col = detect_label_column(train_df)
    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' missing from test parquet")
    print(f"Label column: '{label_col}'")

    model, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  |  token_format: {token_format}  |  vocab: {tokenizer.vocab_size}")

    train_loader, val_loader, test_loader = _make_loaders_for_classification(
        train_df, test_df, tokenizer, label_col, token_format=token_format, device=device
    )

    train_classifier_loop(
        model, train_loader, val_loader, test_loader,
        device, autocast_dtype, model.config.to_dict(), run_name="mode_b",
    )


def run_pretrain_finetune(device, autocast_dtype):
    """MODE_C: MLM pretrain (Stage 1) → LoRA classifier finetune (Stage 2)."""
    print("\n" + "=" * 60)
    print("MODE C: MLM pretrain → LoRA finetune")
    print("=" * 60)

    # ---- Stage 1: MLM pretraining ----
    print(f"\n[Stage 1] MLM pretraining on {PRETRAIN_PARQUET}")
    pretrain_df = pd.read_parquet(PRETRAIN_PARQUET)
    print(f"Pretrain corpus: {len(pretrain_df):,} molecules (labels ignored)")

    tokenizer = build_binary_tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Internal split for MLM val loss monitoring
    pre_train_idx, pre_val_idx = train_test_split(
        np.arange(len(pretrain_df)),
        test_size=MLM_VAL_FRACTION,
        random_state=SEED,
    )
    mlm_train_df = pretrain_df.iloc[pre_train_idx].reset_index(drop=True)
    mlm_val_df = pretrain_df.iloc[pre_val_idx].reset_index(drop=True)

    mlm_train_ds = MolDataset(mlm_train_df, tokenizer, FP_TYPES, label_col=None,
                              max_length=MAX_POSITION_EMBEDDINGS, token_format="binary")
    mlm_val_ds = MolDataset(mlm_val_df, tokenizer, FP_TYPES, label_col=None,
                            max_length=MAX_POSITION_EMBEDDINGS, token_format="binary")

    mlm_collator = MolecularCollator(
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        mlm_probability=MLM_PROBABILITY,
        vocab_size=tokenizer.vocab_size,
    )
    pin_memory = device.type == "cuda"
    mlm_common = dict(num_workers=NUM_WORKERS, pin_memory=pin_memory,
                      persistent_workers=NUM_WORKERS > 0, collate_fn=mlm_collator)
    mlm_train_loader = DataLoader(mlm_train_ds, batch_size=MLM_BATCH_SIZE, shuffle=True, **mlm_common)
    mlm_val_loader = DataLoader(mlm_val_ds, batch_size=MLM_BATCH_SIZE, shuffle=False, **mlm_common)

    mlm_model = build_mlm_from_scratch(tokenizer.vocab_size).to(device)
    print(f"MLM model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")

    mlm_model = mlm_pretrain_loop(mlm_model, mlm_train_loader, mlm_val_loader,
                                  device, autocast_dtype)

    # ---- Stage 2: Classifier finetune with LoRA ----
    print(f"\n[Stage 2] Classifier finetuning on {TRAIN_PARQUET} with LoRA")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")
    print(f"Test:      {len(test_df):,} from {TEST_PARQUET}")

    label_col = detect_label_column(train_df)
    print(f"Label column: '{label_col}'")
    print(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")

    cls_model = build_classifier_from_scratch(tokenizer.vocab_size).to(device)
    copy_encoder_weights(mlm_model, cls_model)
    print("Copied encoder + segment-embedding weights from MLM model into classifier.")

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

    train_loader, val_loader, test_loader = _make_loaders_for_classification(
        train_df, test_df, tokenizer, label_col, token_format="binary", device=device,
    )

    train_classifier_loop(
        cls_model, train_loader, val_loader, test_loader,
        device, autocast_dtype, cls_model.config.to_dict(), run_name="mode_c",
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
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
    else:
        raise ValueError(
            f"Unknown TRAIN_MODE: {TRAIN_MODE!r}. "
            f"Choose from: {MODE_A_FROM_SCRATCH}, {MODE_B_FROM_HF}, {MODE_C_PRETRAIN_FINETUNE}"
        )


if __name__ == "__main__":
    main()
