"""
Tiny end-to-end DELBERT training + evaluation on the 10-molecule example parquet.

This is a SANITY CHECK script, not a real experiment:
  - Trains a small randomly-initialized DELBERT on 10 molecules (CPU-friendly).
  - Synthetic labels (ECFP4 bit-density above/below median) since the demo
    parquet has no real labels.
  - Evaluates on the SAME data — overfitting is expected and the point.
  - Goal: confirm tokenizer + model + training loop all work together.

Run:
    python test_shay_train.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


PARQUET = "data/WDR91_10-examples.parquet"
FP_TYPES = ["ECFP4", "FCFP6", "ATOMPAIR", "TOPTOR"]
NBITS = 2048
SEED = 0
EPOCHS = 15
BATCH_SIZE = 4
LR = 1e-3


def dense_to_sparse(arr):
    arr = np.asarray(arr, dtype=np.int32)
    nz = np.nonzero(arr)[0]
    return nz.tolist(), arr[nz].tolist()


class TinyMolDataset(Dataset):
    """Pre-tokenizes all molecules in __init__ for simplicity."""

    def __init__(self, df, tokenizer, fp_types, labels):
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
                "labels": int(labels[i]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_synthetic_labels(df):
    """Label 1 if ECFP4 bit-density above median, else 0. Provides a learnable signal."""
    density = df["ECFP4"].apply(lambda a: int(np.sum(np.asarray(a) > 0)))
    threshold = density.median()
    return (density > threshold).astype(int).tolist(), threshold


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # 1) Load data
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} molecules from {PARQUET}")

    labels, threshold = make_synthetic_labels(df)
    print(f"Synthetic labels (ECFP4 density > {threshold}): {labels}")

    # 2) Tokenizer (binary vocab — deterministic, no data scan needed)
    vocab_data = build_binary_vocabulary(FP_TYPES, nbits=NBITS)
    tokenizer = create_molecular_tokenizer(
        vocabulary=vocab_data["token_to_id"],
        fingerprint_types=FP_TYPES,
        token_format="binary",
        fingerprint_nbits=NBITS,
    )
    print(f"Vocab size: {tokenizer.vocab_size}")

    # 3) Dataset + collator
    dataset = TinyMolDataset(df, tokenizer, FP_TYPES, labels)
    seq_lens = [len(s["input_ids"]) for s in dataset.samples]
    print(f"Token-sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.0f}")

    collator = MolecularCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    # 4) Tiny model from scratch
    config = DELBERTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
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
    model.train()
    print(f"\n--- Training ({EPOCHS} epochs, batch_size={BATCH_SIZE}) ---")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in loader:
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
        print(f"  Epoch {epoch+1:2d} | loss: {total_loss / len(loader):.4f}")

    # 6) Evaluate on the SAME data (sanity check — should overfit)
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                segment_ids=batch["segment_ids"].to(device),
            )
            probs = F.softmax(outputs["logits"], dim=-1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

    print("\n--- Evaluation on training data (overfitting check) ---")
    print(f"  Labels: {all_labels}")
    print(f"  Preds:  {all_preds}")
    print(f"  Probs:  [{', '.join(f'{p:.3f}' for p in all_probs)}]")
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    print(f"  Accuracy: {correct}/{len(all_labels)} = {correct / len(all_labels):.0%}")


if __name__ == "__main__":
    main()
