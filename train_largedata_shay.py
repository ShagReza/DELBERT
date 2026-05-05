"""
DELBERT classification training on a full-sized dataset with the paper's
standard hyperparameters.

Trains a full-architecture DELBERT classifier (~70M params) from scratch on
`data/train.parquet`, monitors a held-out internal validation split for early
stopping, and evaluates on `data/test.parquet` at the end.

Hyperparameters mirror the DELBERT team's defaults in:
  - configs/experiment/pretrain_example.yaml   (architecture)
  - configs/experiment/classify_example.yaml   (training/loss)

Both parquet files must contain:
  - ECFP4, FCFP6, ATOMPAIR, TOPTOR  (length-2048 dense fingerprint arrays)
  - A binary label column (auto-detected: LABEL / label / ENRICHED / target / active / y)

Run:
    python train_largedata_shay.py
"""

import os

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

from delbert.data.tokenizer import create_molecular_tokenizer
from delbert.data.transforms import (
    MolecularCollator,
    build_binary_vocabulary,
    molecule_to_tokens,
)
from delbert.models.delbert_model import (
    DELBERTConfig,
    DELBERTForSequenceClassification,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PARQUET = "data/train.parquet"
TEST_PARQUET = "data/test.parquet"
CHECKPOINT_DIR = "checkpoints"
PREDICTIONS_CSV = "predictions_test.csv"

FP_TYPES = ["ECFP4", "FCFP6", "ATOMPAIR", "TOPTOR"]
NBITS = 2048

LABEL_CANDIDATES = ["LABEL", "label", "ENRICHED", "enriched", "target", "active", "y"]

# ---------------------------------------------------------------------------
# Architecture (paper defaults: ~70M params)
# Source: configs/experiment/pretrain_example.yaml
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
LOSS_TYPE = "weighted_ce"
POS_CLASS_WEIGHT = 1.0  # train data is balanced; raise (e.g. 15.0) for imbalanced train sets

# ---------------------------------------------------------------------------
# Training (paper defaults)
# ---------------------------------------------------------------------------
SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_CLIP_VAL = 1.0
NUM_WORKERS = 4

# Validation / early stopping
VAL_FRACTION = 0.1
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 0.001
MONITOR_METRIC = "pr_auc"  # 'pr_auc' or 'roc_auc'

# Mixed precision (GPU only — paper uses bf16-mixed)
USE_BF16_ON_GPU = True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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
    """Pre-tokenizes all molecules in __init__."""

    def __init__(self, df, tokenizer, fp_types, label_col, max_length):
        labels = df[label_col].astype(int).tolist()
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
                token_format="binary",
                nbits=NBITS,
            )
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                seg_ids = seg_ids[:max_length]

            self.samples.append({
                "input_ids": input_ids,
                "segment_ids": seg_ids,
                "labels": labels[i],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ---------------------------------------------------------------------------
# Optimizer / scheduler — matches ClassificationModel.configure_optimizers
# ---------------------------------------------------------------------------

def make_optimizer(model, learning_rate, weight_decay):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if any(nd in n for nd in no_decay) else decay).append(p)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # 1) Load
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"Train+val: {len(train_df):,} molecules from {TRAIN_PARQUET}")
    print(f"Test:      {len(test_df):,} molecules from {TEST_PARQUET}")

    label_col = detect_label_column(train_df)
    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' missing from test parquet")
    print(f"Label column: '{label_col}'")
    print(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")
    print(f"Test class balance:      {test_df[label_col].value_counts().to_dict()}")

    # 2) Internal train/val split for early stopping (stratified)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=train_df[label_col].values,
    )
    train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_df_split = train_df.iloc[val_idx].reset_index(drop=True)
    print(f"Internal split: train={len(train_df_split):,}, val={len(val_df_split):,}")

    # 3) Tokenizer (binary vocab)
    vocab_data = build_binary_vocabulary(FP_TYPES, nbits=NBITS)
    tokenizer = create_molecular_tokenizer(
        vocabulary=vocab_data["token_to_id"],
        fingerprint_types=FP_TYPES,
        token_format="binary",
        fingerprint_nbits=NBITS,
    )
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 4) Datasets / loaders
    train_ds = MolDataset(train_df_split, tokenizer, FP_TYPES, label_col, MAX_POSITION_EMBEDDINGS)
    val_ds = MolDataset(val_df_split, tokenizer, FP_TYPES, label_col, MAX_POSITION_EMBEDDINGS)
    test_ds = MolDataset(test_df, tokenizer, FP_TYPES, label_col, MAX_POSITION_EMBEDDINGS)

    seq_lens = [len(s["input_ids"]) for s in train_ds.samples]
    print(f"Train seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

    collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )

    # 5) Full-size model (paper architecture)
    config = DELBERTConfig(
        vocab_size=tokenizer.vocab_size,
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
    model = DELBERTForSequenceClassification(config, num_labels=NUM_LABELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # 6) Loss — weighted CE with pos_class_weight
    class_weights = torch.tensor([1.0, POS_CLASS_WEIGHT], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Loss: weighted CE  weights=[1.0, {POS_CLASS_WEIGHT}]")

    # 7) Optimizer + cosine schedule with warmup
    optim = make_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    print(f"Total steps: {total_steps},  warmup steps: {warmup_steps}")

    # 8) Train loop with early stopping + best-checkpoint tracking
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_metric = -float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    print(f"\n--- Training ({NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}) ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optim.step()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_metrics, *_ = evaluate(model, val_loader, device, autocast_dtype)
        current = val_metrics[MONITOR_METRIC]

        improved = current > best_metric + EARLY_STOP_MIN_DELTA
        if improved:
            best_metric = current
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "val_metrics": val_metrics,
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

    # 9) Load best checkpoint and evaluate on test
    print(f"\n--- Loading best checkpoint (epoch {best_epoch}, val_{MONITOR_METRIC}={best_metric:.4f}) ---")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n--- Final test-set evaluation ---")
    test_metrics, probs, preds, labels = evaluate(model, test_loader, device, autocast_dtype)
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")

    pd.DataFrame({
        "label": labels,
        "pred": preds,
        "prob_active": probs,
    }).to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nSaved test predictions to {PREDICTIONS_CSV}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
