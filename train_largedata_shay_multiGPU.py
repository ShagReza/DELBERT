"""DELBERT multi-GPU DDP training.

Mirrors train_largedata_shay.py (same modes, hyperparameters, outputs) but uses
DistributedDataParallel across multiple GPUs on a single node.

Launch with torchrun:

    torchrun --nproc_per_node=4 train_largedata_shay_multiGPU.py

(replace 4 with the number of GPUs on your instance)

Key DDP differences from the single-GPU script:
  - DistributedSampler shards the training set across ranks.
  - Model is wrapped in DistributedDataParallel after .to(device).
  - Per-rank seed offset for shuffle diversity, but same model init.
  - Validation, test, plots, results.csv, and checkpoint saving are rank-0 only;
    other ranks wait at barriers. The early-stopping decision is broadcast so
    all ranks loop in lockstep.
  - BATCH_SIZE is per-GPU; effective batch = BATCH_SIZE * world_size.

All constants come from train_largedata_shay.py and stay in sync. To change a
hyperparameter, edit it once in that file. (The only DDP-specific knob is the
launch flag --nproc_per_node.)
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup

from delbert.data.transforms import MolecularCollator

# Re-import everything else from the single-GPU script so settings stay in sync.
from train_largedata_shay import (
    # Mode constants & switch
    MODE_A_FROM_SCRATCH, MODE_B_FROM_HF, MODE_C_PRETRAIN_FINETUNE,
    MODE_D_HF_THEN_MLM_FINETUNE, TRAIN_MODE,
    # Run identity & paths
    RUN_NAME, RUNS_DIR,
    TRAIN_PARQUET, TEST_PARQUET, PRETRAIN_PARQUET, HF_MODEL_ID,
    # Fingerprints + labels
    FP_TYPES, NBITS, LABEL_CANDIDATES,
    TOKEN_FORMAT, COUNT_MIN_TOKEN_FREQUENCY,
    # Architecture
    HIDDEN_SIZE, NUM_HIDDEN_LAYERS, NUM_ATTENTION_HEADS, INTERMEDIATE_SIZE,
    MAX_POSITION_EMBEDDINGS, GLOBAL_ROPE_THETA, LOCAL_ATTENTION,
    HIDDEN_DROPOUT_PROB, ATTENTION_DROPOUT_PROB, USE_SEGMENT_EMBEDDINGS,
    SEGMENT_POSITION_ENCODING,
    # Classifier head
    NUM_LABELS, CLASSIFIER_DROPOUT, CLASSIFIER_POOLING, POS_CLASS_WEIGHT,
    # LoRA
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    # Training
    SEED, NUM_EPOCHS, BATCH_SIZE, TEST_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_RATIO, GRADIENT_CLIP_VAL, NUM_WORKERS,
    VAL_FRACTION, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, MONITOR_METRIC,
    MLM_NUM_EPOCHS, MLM_BATCH_SIZE, MLM_LEARNING_RATE,
    MLM_PROBABILITY, MLM_VAL_FRACTION, MLM_CHECKPOINT_TO_LOAD,
    SPAN_SHUFFLE_PROBABILITY,
    USE_BF16_ON_GPU, AREA_HITS_K,
    # Helpers
    detect_label_column, MolDataset, make_optimizer, evaluate_classifier,
    make_diagnostic_plots, compute_test_metrics_table, save_results_csv,
    build_tokenizer_for_mode,
    build_classifier_from_scratch, build_mlm_from_scratch,
    load_hf_classifier, copy_encoder_weights, build_mlm_from_hf_classifier,
    load_mlm_from_checkpoint,
)


# ===========================================================================
# DDP utilities
# ===========================================================================

def init_ddp():
    """Initialize from torchrun env. Returns (local_rank, world_size, rank)."""
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank, dist.get_world_size(), dist.get_rank()
    return 0, 1, 0


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_rank0():
    return not is_distributed() or dist.get_rank() == 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def rprint(*args, **kwargs):
    """Print only on rank 0."""
    if is_rank0():
        print(*args, **kwargs)


def barrier():
    if is_distributed():
        dist.barrier()


# ===========================================================================
# Run setup (rank-0 only)
# ===========================================================================

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def get_run_dir():
    return os.path.join(RUNS_DIR, RUN_NAME)


def _collect_settings():
    return {
        "run_name": RUN_NAME,
        "runs_dir": RUNS_DIR,
        "train_mode": TRAIN_MODE,
        "ddp_world_size": get_world_size(),
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
            "r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES,
        },
        "training": {
            "seed": SEED,
            "num_epochs": NUM_EPOCHS,
            "batch_size_per_gpu": BATCH_SIZE,
            "effective_batch_size": BATCH_SIZE * get_world_size(),
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
            "batch_size_per_gpu": MLM_BATCH_SIZE,
            "effective_batch_size": MLM_BATCH_SIZE * get_world_size(),
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


def setup_run():
    """Rank-0 only: create run dir, write config.json, tee stdout/stderr to log file."""
    if not is_rank0():
        return
    run_dir = get_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(_collect_settings(), f, indent=2)
    log_path = os.path.join(run_dir, "training_log.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Run directory: {run_dir}")
    print(f"World size (GPUs): {get_world_size()}")


# ===========================================================================
# DDP-aware data loaders
# ===========================================================================

def make_classification_loaders(train_df, tokenizer, label_col, token_format, device):
    """Train uses DistributedSampler; val is plain (rank-0 only iterates it).

    The test DataLoader is built later by build_test_loader_from_path() — only
    once training has finished and memory is at its leanest.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=train_df[label_col].values,
    )
    train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_df_split = train_df.iloc[val_idx].reset_index(drop=True)
    rprint(f"Internal split: train={len(train_df_split):,}, val={len(val_df_split):,}")

    max_len = MAX_POSITION_EMBEDDINGS
    train_ds = MolDataset(train_df_split, tokenizer, FP_TYPES, label_col, max_len, token_format)
    val_ds = MolDataset(val_df_split, tokenizer, FP_TYPES, label_col, max_len, token_format)

    seq_lens = [len(s["input_ids"]) for s in train_ds.samples]
    rprint(f"Train seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

    # Span shuffling is only applied during training; eval stays deterministic.
    train_collator = MolecularCollator(
        pad_token_id=tokenizer.pad_token_id,
        span_shuffle_probability=SPAN_SHUFFLE_PROBABILITY,
    )
    eval_collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    pin_memory = device.type == "cuda"

    if is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=train_collator,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=eval_collator,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader, train_sampler


def build_test_loader_from_path(test_parquet_path, tokenizer, label_col, token_format, device):
    """Read the test parquet (column-filtered) and build a DataLoader for it.

    Called after classifier training completes — see _make_loaders_for_classification
    docstring in train_largedata_shay.py for the deferred-read rationale.
    """
    cols_needed = FP_TYPES + [label_col]
    rprint(f"Reading {test_parquet_path} (columns: {cols_needed})...")
    test_df = pd.read_parquet(test_parquet_path, columns=cols_needed)
    rprint(f"Test:      {len(test_df):,} from {test_parquet_path}")
    rprint(f"Test class balance: {test_df[label_col].value_counts().to_dict()}")

    test_ds = MolDataset(
        test_df, tokenizer, FP_TYPES, label_col,
        MAX_POSITION_EMBEDDINGS, token_format,
    )
    eval_collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    pin_memory = device.type == "cuda"
    rprint(f"Test DataLoader batch_size={TEST_BATCH_SIZE} (inference-only).")
    return DataLoader(
        test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
        collate_fn=eval_collator, num_workers=NUM_WORKERS,
        pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )


# ===========================================================================
# DDP-aware classifier training loop
# ===========================================================================

def train_classifier_loop(model, train_loader, val_loader, test_loader_builder,
                          train_sampler, device, autocast_dtype, config_dict,
                          run_name="run"):
    class_weights = torch.tensor([1.0, POS_CLASS_WEIGHT], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    rprint(f"Loss: weighted CE  weights=[1.0, {POS_CLASS_WEIGHT}]")

    if is_distributed():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optim = make_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    rprint(f"Total steps: {total_steps},  warmup steps: {warmup_steps}")

    best_metric = -float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    best_ckpt_path = None
    if is_rank0():
        run_dir = get_run_dir()
        os.makedirs(run_dir, exist_ok=True)
        best_ckpt_path = os.path.join(run_dir, f"{run_name}_best.pt")

    rprint(
        f"\n--- Classifier training ({NUM_EPOCHS} epochs, "
        f"per-GPU batch={BATCH_SIZE}, effective batch={BATCH_SIZE * get_world_size()}, "
        f"lr={LEARNING_RATE}) ---"
    )
    for epoch in range(1, NUM_EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            segment_ids = batch["segment_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                segment_ids=segment_ids)
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

        # Val: rank 0 only; broadcast the monitored metric so all ranks agree on early stop.
        barrier()
        val_metrics = None
        if is_rank0():
            base_model = model.module if hasattr(model, "module") else model
            val_metrics, *_ = evaluate_classifier(base_model, val_loader, device, autocast_dtype)
            current = val_metrics[MONITOR_METRIC]
        else:
            current = -float("inf")

        if is_distributed():
            t = torch.tensor([current], dtype=torch.float64, device=device)
            dist.broadcast(t, src=0)
            current = float(t.item())

        improved = current > best_metric + EARLY_STOP_MIN_DELTA
        if improved:
            best_metric = current
            best_epoch = epoch
            epochs_without_improve = 0
            if is_rank0():
                base_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "config": config_dict,
                    "val_metrics": val_metrics,
                    "train_mode": TRAIN_MODE,
                }, best_ckpt_path)
        else:
            epochs_without_improve += 1

        if is_rank0():
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
            rprint(
                f"  Early stopping at epoch {epoch} (best: epoch {best_epoch}, "
                f"val_{MONITOR_METRIC}={best_metric:.4f})"
            )
            break
        barrier()

    # Final test eval — rank 0 only writes artifacts.
    barrier()
    if is_rank0():
        rprint(
            f"\n--- Loading best checkpoint (epoch {best_epoch}, "
            f"val_{MONITOR_METRIC}={best_metric:.4f}) ---"
        )
        base_model = model.module if hasattr(model, "module") else model
        ckpt = torch.load(best_ckpt_path, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])

        # Build test loader NOW — training done, optimizer state gone, memory leanest.
        rprint("\n--- Building test loader (deferred read) ---")
        test_loader = test_loader_builder()

        rprint("\n--- Final test-set evaluation ---")
        test_metrics, probs, preds, labels = evaluate_classifier(
            base_model, test_loader, device, autocast_dtype
        )
        for k, v in test_metrics.items():
            print(f"  test_{k}: {v:.4f}")

        run_dir = get_run_dir()
        predictions_path = os.path.join(run_dir, "predictions_test.csv")
        pd.DataFrame({"label": labels, "pred": preds, "prob_active": probs}).to_csv(
            predictions_path, index=False
        )
        print(f"\nSaved test predictions to {predictions_path}")
        print(f"Best checkpoint: {best_ckpt_path}")

        print("\n--- Generating diagnostic plots ---")
        make_diagnostic_plots(labels, probs, run_dir, title=RUN_NAME)

        print("\n--- Computing test metrics table ---")
        metrics_table = compute_test_metrics_table(labels, preds, probs, K=AREA_HITS_K)
        save_results_csv(metrics_table, run_dir)

    barrier()


# ===========================================================================
# DDP-aware MLM pretrain loop
# ===========================================================================

def mlm_pretrain_loop(model, train_loader, train_sampler, val_loader,
                     device, autocast_dtype):
    if is_distributed():
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optim = make_optimizer(model, MLM_LEARNING_RATE, WEIGHT_DECAY)
    total_steps = MLM_NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    rprint(f"MLM total steps: {total_steps},  warmup steps: {warmup_steps}")

    rprint(
        f"\n--- MLM pretraining ({MLM_NUM_EPOCHS} epochs, "
        f"per-GPU batch={MLM_BATCH_SIZE}, effective batch={MLM_BATCH_SIZE * get_world_size()}, "
        f"lr={MLM_LEARNING_RATE}) ---"
    )
    for epoch in range(1, MLM_NUM_EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            segment_ids = batch["segment_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
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

        # Val MLM loss — rank 0 only.
        barrier()
        if is_rank0():
            base_model = model.module if hasattr(model, "module") else model
            base_model.eval()
            val_loss, val_n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    iid = batch["input_ids"].to(device, non_blocking=True)
                    am = batch["attention_mask"].to(device, non_blocking=True)
                    sid = batch["segment_ids"].to(device, non_blocking=True)
                    lab = batch["labels"].to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                        enabled=autocast_dtype is not None):
                        out = base_model(input_ids=iid, attention_mask=am,
                                         segment_ids=sid, labels=lab)
                    val_loss += out["loss"].item()
                    val_n += 1
            val_loss /= max(val_n, 1)
            print(f"  MLM Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        barrier()

    # Save MLM-pretrained encoder — rank 0 only.
    if is_rank0():
        base_model = model.module if hasattr(model, "module") else model
        run_dir = get_run_dir()
        os.makedirs(run_dir, exist_ok=True)
        mlm_ckpt_path = os.path.join(run_dir, "mlm_pretrained.pt")
        torch.save({
            "model_state_dict": base_model.state_dict(),
            "config": base_model.config.to_dict(),
        }, mlm_ckpt_path)
        print(f"\nMLM checkpoint saved to {mlm_ckpt_path}")
    barrier()
    return model.module if hasattr(model, "module") else model


# ===========================================================================
# Mode runners
# ===========================================================================

def run_from_scratch(device, autocast_dtype):
    rprint("\n" + "=" * 60)
    rprint("MODE A: From scratch (no pretraining) - multi-GPU DDP")
    rprint("=" * 60)

    train_df = pd.read_parquet(TRAIN_PARQUET)
    rprint(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    rprint(f"Label column: '{label_col}'")
    rprint(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")

    tokenizer = build_tokenizer_for_mode(df_for_count_vocab=train_df)
    rprint(f"Vocab size: {tokenizer.vocab_size}  |  token_format: {TOKEN_FORMAT}")

    train_loader, val_loader, train_sampler = make_classification_loaders(
        train_df, tokenizer, label_col, token_format=TOKEN_FORMAT, device=device,
    )

    model = build_classifier_from_scratch(tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rprint(f"Model parameters: {n_params:,}")

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, TOKEN_FORMAT, device,
        )

    train_classifier_loop(
        model, train_loader, val_loader, _test_loader_builder, train_sampler,
        device, autocast_dtype, model.config.to_dict(), run_name="mode_a",
    )


def run_from_hf(device, autocast_dtype):
    rprint("\n" + "=" * 60)
    rprint(f"MODE B: From HuggingFace ({HF_MODEL_ID}) - multi-GPU DDP")
    rprint("=" * 60)

    train_df = pd.read_parquet(TRAIN_PARQUET)
    rprint(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    rprint(f"Label column: '{label_col}'")

    # Rank 0 downloads first; others reuse the on-disk cache (single-node Workbench).
    if is_rank0():
        model, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)
    barrier()
    if not is_rank0():
        model, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rprint(
        f"Model parameters: {n_params:,}  |  token_format: {token_format}  "
        f"|  vocab: {tokenizer.vocab_size}"
    )

    train_loader, val_loader, train_sampler = make_classification_loaders(
        train_df, tokenizer, label_col, token_format=token_format, device=device,
    )

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, token_format, device,
        )

    train_classifier_loop(
        model, train_loader, val_loader, _test_loader_builder, train_sampler,
        device, autocast_dtype, model.config.to_dict(), run_name="mode_b",
    )


def run_pretrain_finetune(device, autocast_dtype):
    rprint("\n" + "=" * 60)
    rprint("MODE C: MLM pretrain -> LoRA finetune - multi-GPU DDP")
    rprint("=" * 60)

    # ---- Stage 1: MLM pretraining (or resume from checkpoint) ----
    pretrain_df = pd.read_parquet(PRETRAIN_PARQUET)
    rprint(f"Pretrain corpus: {len(pretrain_df):,} molecules (labels ignored)")
    tokenizer = build_tokenizer_for_mode(df_for_count_vocab=pretrain_df)
    rprint(f"Vocab size: {tokenizer.vocab_size}  |  token_format: {TOKEN_FORMAT}")
    import gc

    if MLM_CHECKPOINT_TO_LOAD:
        rprint(f"\n[Stage 1] SKIPPED — loading MLM checkpoint from {MLM_CHECKPOINT_TO_LOAD}")
        mlm_model = load_mlm_from_checkpoint(MLM_CHECKPOINT_TO_LOAD, device)
        rprint(f"MLM model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")
        del pretrain_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        rprint(f"\n[Stage 1] MLM pretraining on {PRETRAIN_PARQUET}")

        pre_train_idx, pre_val_idx = train_test_split(
            np.arange(len(pretrain_df)), test_size=MLM_VAL_FRACTION, random_state=SEED,
        )
        mlm_train_df = pretrain_df.iloc[pre_train_idx].reset_index(drop=True)
        mlm_val_df = pretrain_df.iloc[pre_val_idx].reset_index(drop=True)

        mlm_train_ds = MolDataset(
            mlm_train_df, tokenizer, FP_TYPES, label_col=None,
            max_length=MAX_POSITION_EMBEDDINGS, token_format=TOKEN_FORMAT,
        )
        mlm_val_ds = MolDataset(
            mlm_val_df, tokenizer, FP_TYPES, label_col=None,
            max_length=MAX_POSITION_EMBEDDINGS, token_format=TOKEN_FORMAT,
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

        if is_distributed():
            mlm_train_sampler = DistributedSampler(mlm_train_ds, shuffle=True, drop_last=False)
        else:
            mlm_train_sampler = None
        mlm_train_loader = DataLoader(
            mlm_train_ds, batch_size=MLM_BATCH_SIZE, sampler=mlm_train_sampler,
            shuffle=(mlm_train_sampler is None), collate_fn=mlm_train_collator,
            num_workers=NUM_WORKERS, pin_memory=pin_memory,
            persistent_workers=NUM_WORKERS > 0,
        )
        mlm_val_loader = DataLoader(
            mlm_val_ds, batch_size=MLM_BATCH_SIZE, shuffle=False,
            collate_fn=mlm_eval_collator, num_workers=NUM_WORKERS,
            pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
        )

        mlm_model = build_mlm_from_scratch(tokenizer.vocab_size).to(device)
        rprint(f"MLM model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")

        mlm_model = mlm_pretrain_loop(
            mlm_model, mlm_train_loader, mlm_train_sampler, mlm_val_loader,
            device, autocast_dtype,
        )

        # Free Stage-1 resources before Stage 2 reads the test parquet.
        del mlm_train_loader, mlm_val_loader, mlm_train_ds, mlm_val_ds
        del mlm_train_collator, mlm_eval_collator
        del pretrain_df, mlm_train_df, mlm_val_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rprint("Stage 1 resources released.")

    # ---- Stage 2: Classifier finetune with LoRA ----
    rprint(f"\n[Stage 2] Classifier finetuning on {TRAIN_PARQUET} with LoRA")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    rprint(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    rprint(f"Label column: '{label_col}'")
    rprint(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")

    cls_model = build_classifier_from_scratch(tokenizer.vocab_size).to(device)
    copy_encoder_weights(mlm_model, cls_model)
    rprint("Copied encoder + segment-embedding weights from MLM model into classifier.")

    # MLM model's job is done — its weights are now in cls_model.encoder.
    del mlm_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from delbert.models.finetuning_strategies import get_finetuning_strategy
    strategy = get_finetuning_strategy(
        "lora",
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    strategy.apply(cls_model)

    total = sum(p.numel() for p in cls_model.parameters())
    trainable = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    rprint(
        f"Classifier parameters: total={total:,}  trainable={trainable:,}  "
        f"({100 * trainable / total:.2f}% via LoRA)"
    )

    train_loader, val_loader, train_sampler = make_classification_loaders(
        train_df, tokenizer, label_col, token_format=TOKEN_FORMAT, device=device,
    )

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, TOKEN_FORMAT, device,
        )

    train_classifier_loop(
        cls_model, train_loader, val_loader, _test_loader_builder, train_sampler,
        device, autocast_dtype, cls_model.config.to_dict(), run_name="mode_c",
    )


def run_hf_then_mlm_finetune(device, autocast_dtype):
    """MODE_D: HF weights -> continued MLM on PRETRAIN_PARQUET -> LoRA finetune. DDP version."""
    rprint("\n" + "=" * 60)
    rprint(f"MODE D: HF ({HF_MODEL_ID}) -> MLM on PRETRAIN_PARQUET -> LoRA finetune (DDP)")
    rprint("=" * 60)

    # ---- Load HF model on rank 0 first; cache shared across ranks ----
    if is_rank0():
        hf_cls_model, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)
    barrier()
    if not is_rank0():
        hf_cls_model, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)

    rprint(f"Tokenizer vocab: {tokenizer.vocab_size}  |  token_format: {token_format}")

    # ---- Stage 1: continued MLM on PRETRAIN_PARQUET ----
    rprint(f"\n[Stage 1] MLM pretraining on {PRETRAIN_PARQUET}")
    rprint("Note: starting from HF-pretrained encoder. Consider lowering")
    rprint("      MLM_LEARNING_RATE (e.g. 5e-5) and MLM_NUM_EPOCHS (e.g. 10-20).")
    pretrain_df = pd.read_parquet(PRETRAIN_PARQUET)
    rprint(f"Pretrain corpus: {len(pretrain_df):,} molecules (labels ignored)")

    mlm_model = build_mlm_from_hf_classifier(hf_cls_model).to(device)
    rprint(f"MLM model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")

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

    if is_distributed():
        mlm_train_sampler = DistributedSampler(mlm_train_ds, shuffle=True, drop_last=False)
    else:
        mlm_train_sampler = None
    mlm_train_loader = DataLoader(
        mlm_train_ds, batch_size=MLM_BATCH_SIZE, sampler=mlm_train_sampler,
        shuffle=(mlm_train_sampler is None), collate_fn=mlm_train_collator,
        num_workers=NUM_WORKERS, pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )
    mlm_val_loader = DataLoader(
        mlm_val_ds, batch_size=MLM_BATCH_SIZE, shuffle=False,
        collate_fn=mlm_eval_collator, num_workers=NUM_WORKERS,
        pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )

    mlm_model = mlm_pretrain_loop(
        mlm_model, mlm_train_loader, mlm_train_sampler, mlm_val_loader,
        device, autocast_dtype,
    )

    # ---- Stage 2: LoRA classifier finetune on TRAIN_PARQUET ----
    rprint(f"\n[Stage 2] Classifier finetuning on {TRAIN_PARQUET} with LoRA")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    rprint(f"Train+val: {len(train_df):,} from {TRAIN_PARQUET}")

    label_col = detect_label_column(train_df)
    rprint(f"Label column: '{label_col}'")
    rprint(f"Train+val class balance: {train_df[label_col].value_counts().to_dict()}")

    from delbert.models.delbert_model import DELBERTForSequenceClassification
    cls_model = DELBERTForSequenceClassification(mlm_model.config, num_labels=NUM_LABELS).to(device)
    copy_encoder_weights(mlm_model, cls_model)
    rprint("Copied encoder + segment-embedding weights from MLM model into classifier.")

    from delbert.models.finetuning_strategies import get_finetuning_strategy
    strategy = get_finetuning_strategy(
        "lora",
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    strategy.apply(cls_model)

    total = sum(p.numel() for p in cls_model.parameters())
    trainable = sum(p.numel() for p in cls_model.parameters() if p.requires_grad)
    rprint(
        f"Classifier parameters: total={total:,}  trainable={trainable:,}  "
        f"({100 * trainable / total:.2f}% via LoRA)"
    )

    train_loader, val_loader, train_sampler = make_classification_loaders(
        train_df, tokenizer, label_col, token_format=token_format, device=device,
    )

    def _test_loader_builder():
        return build_test_loader_from_path(
            TEST_PARQUET, tokenizer, label_col, token_format, device,
        )

    train_classifier_loop(
        cls_model, train_loader, val_loader, _test_loader_builder, train_sampler,
        device, autocast_dtype, cls_model.config.to_dict(), run_name="mode_d",
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    local_rank, world_size, rank = init_ddp()
    setup_run()

    # Same SEED for model init across ranks; different shuffle stream per rank.
    torch.manual_seed(SEED)
    np.random.seed(SEED + rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and USE_BF16_ON_GPU) else None
    rprint(f"Device: {device}  |  autocast: {autocast_dtype}  |  world_size: {world_size}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    rprint(f"TRAIN_MODE: {TRAIN_MODE}")
    barrier()

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
            f"Unknown TRAIN_MODE: {TRAIN_MODE!r}. Choose: "
            f"{MODE_A_FROM_SCRATCH}, {MODE_B_FROM_HF}, "
            f"{MODE_C_PRETRAIN_FINETUNE}, {MODE_D_HF_THEN_MLM_FINETUNE}"
        )

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
