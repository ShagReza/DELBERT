"""
DELBERT training + evaluation on local parquet files.

Trains a small (CPU-friendly) DELBERT classifier from scratch on
`data/train.parquet` and evaluates on `data/test.parquet`.

Both parquet files must contain:
  - ECFP4, FCFP6, ATOMPAIR, TOPTOR  (length-2048 dense fingerprint arrays)
  - A binary label column (auto-detected: LABEL / label / ENRICHED / target / active / y)

Run:
    python train_shay.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

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


TRAIN_PARQUET = "data/train.parquet"
TEST_PARQUET = "data/test.parquet"
FP_TYPES = ["ECFP4", "FCFP6", "ATOMPAIR", "TOPTOR"]
NBITS = 2048

LABEL_CANDIDATES = ["LABEL", "label", "ENRICHED", "enriched", "target", "active", "y"]

SEED = 0
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-3


def detect_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"No label column found in parquet. Tried {LABEL_CANDIDATES}. "
        f"Columns present: {list(df.columns)}"
    )


def dense_to_sparse(arr):
    arr = np.asarray(arr, dtype=np.int32)
    nz = np.nonzero(arr)[0]
    return nz.tolist(), arr[nz].tolist()


class MolDataset(Dataset):
    """Pre-tokenizes all molecules in __init__."""

    def __init__(self, df, tokenizer, fp_types, label_col):
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

            self.samples.append({
                "input_ids": input_ids,
                "segment_ids": seg_ids,
                "labels": labels[i],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                segment_ids=batch["segment_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs["loss"]
            probs = F.softmax(outputs["logits"], dim=-1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            total_loss += loss.item()
            n_batches += 1

    metrics = {"loss": total_loss / max(n_batches, 1)}
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)
    if len(set(all_labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
        metrics["pr_auc"] = average_precision_score(all_labels, all_probs)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics, all_probs, all_preds, all_labels


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Load
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    print(f"Train: {len(train_df)} molecules from {TRAIN_PARQUET}")
    print(f"Test:  {len(test_df)} molecules from {TEST_PARQUET}")

    label_col = detect_label_column(train_df)
    if label_col not in test_df.columns:
        raise ValueError(
            f"Label column '{label_col}' found in train but missing from test. "
            f"Test columns: {list(test_df.columns)}"
        )
    print(f"Label column: '{label_col}'")
    print(f"Train class balance: {train_df[label_col].value_counts().to_dict()}")
    print(f"Test class balance:  {test_df[label_col].value_counts().to_dict()}")

    # 2) Tokenizer (binary vocab — deterministic, no data scan)
    vocab_data = build_binary_vocabulary(FP_TYPES, nbits=NBITS)
    tokenizer = create_molecular_tokenizer(
        vocabulary=vocab_data["token_to_id"],
        fingerprint_types=FP_TYPES,
        token_format="binary",
        fingerprint_nbits=NBITS,
    )
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 3) Datasets
    train_ds = MolDataset(train_df, tokenizer, FP_TYPES, label_col)
    test_ds = MolDataset(test_df, tokenizer, FP_TYPES, label_col)

    seq_lens = [len(s["input_ids"]) for s in train_ds.samples]
    print(f"Train seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

    collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    # 4) Small model from scratch (CPU-friendly; bump up if you have GPU)
    config = DELBERTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=1024,
        global_rope_theta=160000.0,
        local_attention=128,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        classifier_dropout=0.1,
        classifier_pooling="cls",
        use_segment_embeddings=True,
    )
    model = DELBERTForSequenceClassification(config, num_labels=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # 5) Train
    optim = AdamW(model.parameters(), lr=LR)
    print(f"\n--- Training ({EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}) ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                segment_ids=batch["segment_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs["loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_metrics, *_ = evaluate(model, test_loader, device)
        print(
            f"  Epoch {epoch+1:2d} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={val_metrics['loss']:.4f} | "
            f"test_acc={val_metrics['accuracy']:.3f} | "
            f"test_roc_auc={val_metrics['roc_auc']:.3f} | "
            f"test_pr_auc={val_metrics['pr_auc']:.3f}"
        )

    # 6) Final evaluation + save predictions
    print("\n--- Final test-set evaluation ---")
    final_metrics, probs, preds, labels = evaluate(model, test_loader, device)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    out_csv = "predictions_test.csv"
    pd.DataFrame({
        "label": labels,
        "pred": preds,
        "prob_active": probs,
    }).to_csv(out_csv, index=False)
    print(f"\nSaved test predictions to {out_csv}")


if __name__ == "__main__":
    main()
