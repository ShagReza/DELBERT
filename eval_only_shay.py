"""eval_only_shay.py — Standalone test-set evaluation for DELBERT classifier checkpoints.

Loads a saved classifier checkpoint (from a Mode A/B/C/D training run), runs
inference on a test parquet, and writes predictions + diagnostic plots +
results.csv to runs/<RUN_NAME>/.

Supports any of the four training modes' checkpoints — LoRA adapters are
auto-detected and applied before loading weights.

Edit the constants at the top, then:
    python eval_only_shay.py

Outputs written to runs/<RUN_NAME>/:
    eval_config.json          — snapshot of the constants used for this eval
    predictions_test.csv      — per-molecule (label, pred, prob_active)
    hit_curve.png             — cumulative hit curve
    enrichment_curve.png      — fold enrichment vs top-n
    pvalue_curve.png          — hypergeometric -log10(p) vs top-n
    results.csv               — Test_* metric table
"""

import gc
import json
import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

# Reuse everything from the training script so behavior stays in sync
from delbert.data.transforms import MolecularCollator
from delbert.models.delbert_model import (
    DELBERTConfig,
    DELBERTForSequenceClassification,
)
from train_largedata_shay import (
    # Constants used by data prep / eval
    FP_TYPES, NBITS, MAX_POSITION_EMBEDDINGS, AREA_HITS_K,
    USE_BF16_ON_GPU, NUM_WORKERS, LABEL_CANDIDATES,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    # Tokenizer builders
    build_binary_tokenizer, build_count_tokenizer, load_hf_classifier,
    # Dataset + eval helpers
    MolDataset, evaluate_classifier,
    make_diagnostic_plots, compute_test_metrics_table, save_results_csv,
)


# ===========================================================================
# Config — edit these for your eval run
# ===========================================================================

# Required: where the test parquet and checkpoint live, and a name for outputs
TEST_PARQUET = "data/test.parquet"
CHECKPOINT_PATH = "runs/default/mode_b_best.pt"
RUN_NAME = "eval_only"
RUNS_DIR = "runs"

# Tokenizer source — must match how the checkpoint was trained:
#   "binary"           Mode A or C trained with TOKEN_FORMAT="binary"
#                      (deterministic, no extra config needed)
#   "count_from_data"  Mode A or C trained with TOKEN_FORMAT="count"
#                      (rebuilds vocab from SOURCE_PARQUET — must be the same
#                      file used during training)
#   "hf"               Mode B or D (uses the HuggingFace tokenizer +
#                      count format from HF_MODEL_ID)
TOKENIZER_SOURCE = "hf"
HF_MODEL_ID = "wanglab/delbert-wdr91"
SOURCE_PARQUET = "data/train.parquet"

# Inference batch — can be larger than training batch (no gradients)
TEST_BATCH_SIZE_LOCAL = 128


# ===========================================================================
# Helpers
# ===========================================================================

def _get_run_dir() -> str:
    return os.path.join(RUNS_DIR, RUN_NAME)


def _detect_label_column_from_parquet(parquet_path: str) -> str:
    """Find the label column without reading any data (uses parquet schema)."""
    schema = pq.read_schema(parquet_path)
    cols = set(schema.names)
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(
        f"No label column found in {parquet_path}. "
        f"Tried {LABEL_CANDIDATES}. Got: {sorted(cols)}"
    )


def _build_tokenizer():
    """Build the same tokenizer that was used during training. Returns (tokenizer, token_format)."""
    if TOKENIZER_SOURCE == "binary":
        print("Building binary tokenizer (deterministic).")
        return build_binary_tokenizer(), "binary"
    if TOKENIZER_SOURCE == "count_from_data":
        print(f"Building count tokenizer from {SOURCE_PARQUET}.")
        cols_needed = list(FP_TYPES)
        df = pd.read_parquet(SOURCE_PARQUET, columns=cols_needed)
        tokenizer = build_count_tokenizer(df)
        del df
        gc.collect()
        return tokenizer, "count"
    if TOKENIZER_SOURCE == "hf":
        print(f"Loading tokenizer from HuggingFace: {HF_MODEL_ID}.")
        _, tokenizer, token_format = load_hf_classifier(HF_MODEL_ID)
        return tokenizer, token_format
    raise ValueError(
        f"Unknown TOKENIZER_SOURCE: {TOKENIZER_SOURCE!r}. "
        f"Choose 'binary', 'count_from_data', or 'hf'."
    )


def _load_checkpoint(ckpt_path: str, device) -> DELBERTForSequenceClassification:
    """Load a classifier checkpoint, applying LoRA structure first if detected."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Reconstruct config (strip non-config fields some serializers add)
    config_dict = dict(ckpt["config"])
    for key in ("architectures", "model_type", "transformers_version", "lora_merged"):
        config_dict.pop(key, None)
    config = DELBERTConfig(**config_dict)

    # Auto-detect num_labels from classifier head weights
    state_dict = ckpt["model_state_dict"]
    num_labels = 2
    for k, v in state_dict.items():
        if k.endswith("classifier.weight"):
            num_labels = v.shape[0]
            break

    train_mode = ckpt.get("train_mode", "unknown")
    print(
        f"Checkpoint info: train_mode={train_mode}, "
        f"hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
        f"vocab={config.vocab_size}, num_labels={num_labels}"
    )

    model = DELBERTForSequenceClassification(config, num_labels=num_labels)

    # Detect LoRA — if any key has 'lora_' or 'base_layer', apply LoRA structure first
    has_lora = any(("lora_" in k or "base_layer" in k) for k in state_dict.keys())
    if has_lora:
        print("Detected LoRA adapters in checkpoint. Applying LoRA structure before loading weights.")
        from delbert.models.finetuning_strategies import get_finetuning_strategy
        strategy = get_finetuning_strategy(
            "lora",
            r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES,
        )
        strategy.apply(model)
    else:
        print("No LoRA adapters detected — loading as a standard classifier.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys when loading state_dict.")
        print(f"  First few: {missing[:3]}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys when loading state_dict.")
        print(f"  First few: {unexpected[:3]}")

    val_metrics = ckpt.get("val_metrics", {})
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded weights from epoch {epoch} (val_metrics: {val_metrics})")

    return model.to(device), config


# ===========================================================================
# Main
# ===========================================================================

def main():
    run_dir = _get_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # Snapshot the eval config so the run folder is self-describing
    settings = {
        "test_parquet": TEST_PARQUET,
        "checkpoint_path": CHECKPOINT_PATH,
        "run_name": RUN_NAME,
        "tokenizer_source": TOKENIZER_SOURCE,
        "hf_model_id": HF_MODEL_ID,
        "source_parquet": SOURCE_PARQUET,
        "fp_types": FP_TYPES,
        "nbits": NBITS,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "test_batch_size": TEST_BATCH_SIZE_LOCAL,
        "num_workers": NUM_WORKERS,
        "use_bf16_on_gpu": USE_BF16_ON_GPU,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "eval_config.json"), "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Run directory: {run_dir}")
    print(f"Eval config snapshot: {os.path.join(run_dir, 'eval_config.json')}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and USE_BF16_ON_GPU) else None
    print(f"Device: {device}  |  autocast: {autocast_dtype}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Build tokenizer (must match the one used during training)
    tokenizer, token_format = _build_tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}  |  token_format: {token_format}")

    # Load checkpoint
    model, config = _load_checkpoint(CHECKPOINT_PATH, device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Sanity check: tokenizer and model vocab must agree
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"Vocab size mismatch: tokenizer={tokenizer.vocab_size}, "
            f"checkpoint expects {config.vocab_size}. "
            f"TOKENIZER_SOURCE / FP_TYPES / HF_MODEL_ID may not match the training config."
        )

    # Detect label column from parquet schema (no data read)
    label_col = _detect_label_column_from_parquet(TEST_PARQUET)
    print(f"Label column: '{label_col}'")

    # Build test loader — column-filtered read keeps RAM low
    cols_needed = FP_TYPES + [label_col]
    print(f"Reading {TEST_PARQUET} (columns: {cols_needed})...")
    test_df = pd.read_parquet(TEST_PARQUET, columns=cols_needed)
    print(f"Test: {len(test_df):,} from {TEST_PARQUET}")
    print(f"Test class balance: {test_df[label_col].value_counts().to_dict()}")

    test_ds = MolDataset(
        test_df, tokenizer, FP_TYPES, label_col,
        MAX_POSITION_EMBEDDINGS, token_format,
    )
    del test_df
    gc.collect()

    eval_collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    pin_memory = device.type == "cuda"
    test_loader = DataLoader(
        test_ds, batch_size=TEST_BATCH_SIZE_LOCAL, shuffle=False,
        collate_fn=eval_collator, num_workers=NUM_WORKERS,
        pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0,
    )
    print(f"Test DataLoader batch_size={TEST_BATCH_SIZE_LOCAL} (inference-only).")

    # Run evaluation
    print("\n--- Running evaluation ---")
    test_metrics, probs, preds, labels = evaluate_classifier(
        model, test_loader, device, autocast_dtype,
    )
    for k, v in test_metrics.items():
        print(f"  test_{k}: {v:.4f}")

    # Save predictions
    pred_path = os.path.join(run_dir, "predictions_test.csv")
    pd.DataFrame({"label": labels, "pred": preds, "prob_active": probs}).to_csv(
        pred_path, index=False,
    )
    print(f"\nSaved test predictions to {pred_path}")

    # Diagnostic plots
    print("\n--- Generating diagnostic plots ---")
    make_diagnostic_plots(labels, probs, run_dir, title=RUN_NAME)

    # Test metrics table
    print("\n--- Computing test metrics table ---")
    metrics_table = compute_test_metrics_table(labels, preds, probs, K=AREA_HITS_K)
    save_results_csv(metrics_table, run_dir)

    print(f"\nDone. All outputs in {run_dir}.")


if __name__ == "__main__":
    main()
