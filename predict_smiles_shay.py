"""predict_smiles_shay.py — SMILES → DELBERT predictions.

Takes a list of SMILES strings, computes the 4 fingerprints DELBERT expects
(ECFP4, FCFP6, ATOMPAIR, TOPTOR) with RDKit, and prints P(active) for each
molecule using a published DELBERT model from HuggingFace.

Edit the SMILES list and MODEL constants at the top, then:

    python predict_smiles_shay.py

Requires:  rdkit  (pip install rdkit)
"""

# ===========================================================================
# Edit these
# ===========================================================================

# List of SMILES strings to predict
SMILES = [
    "CCO",                                      # ethanol (toy example)
    "c1ccccc1",                                 # benzene
    "CC(=O)Oc1ccccc1C(=O)O",                    # aspirin
]

# Published HuggingFace model. Choose by target:
#   "wanglab/delbert-wdr91"   — WDR91
#   "wanglab/delbert-lrrk2"   — LRRK2
#   "wanglab/delbert-setdb1"  — SETDB1
#   "wanglab/delbert-dcaf7"   — DCAF7
MODEL = "wanglab/delbert-wdr91"

# Where to write the predictions CSV (set to "" to skip writing)
OUTPUT_CSV = "smiles_predictions.csv"

# Fingerprint size — paper uses 2048; do not change unless you know why
NBITS = 2048

# Inference settings
BATCH_SIZE = 100     # molecules per forward pass
DEVICE = None        # None = auto-detect; or "cuda" / "cpu"


# ===========================================================================
# Code
# ===========================================================================

import sys

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print(
        "ERROR: rdkit is not installed.  Install with:\n"
        "    pip install rdkit"
    )
    sys.exit(1)

import pandas as pd

from inference.predict import predict


def smiles_to_dense_fps(smiles: str, nbits: int = NBITS):
    """Convert one SMILES → dict of 4 dense count-fingerprint arrays (length nbits each).

    Returns None if RDKit can't parse the SMILES.

    Fingerprint conventions match the published DELBERT models:
      - ECFP4    = Morgan, radius 2
      - FCFP6    = Morgan, radius 3, with feature atom invariants (pharmacophore-style)
      - ATOMPAIR = Hashed atom-pair fingerprint
      - TOPTOR   = Hashed topological-torsion fingerprint
    All are folded to nbits and returned as integer count vectors.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    def _to_dense(fp):
        arr = [0] * nbits
        for bit, count in fp.GetNonzeroElements().items():
            arr[bit] = count
        return arr

    return {
        "ECFP4":    _to_dense(AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=nbits)),
        "FCFP6":    _to_dense(AllChem.GetHashedMorganFingerprint(mol, radius=3, nBits=nbits, useFeatures=True)),
        "ATOMPAIR": _to_dense(AllChem.GetHashedAtomPairFingerprint(mol, nBits=nbits)),
        "TOPTOR":   _to_dense(AllChem.GetHashedTopologicalTorsionFingerprint(mol, nBits=nbits)),
    }


def main():
    print(f"Model:       {MODEL}")
    print(f"Input:       {len(SMILES)} SMILES")
    print(f"FP size:     {NBITS} bits × 4 types (ECFP4, FCFP6, ATOMPAIR, TOPTOR)")
    print()

    # 1) SMILES → fingerprints (RDKit), tracking valid / invalid
    print("Computing fingerprints with RDKit...")
    molecules = []
    valid_smiles = []
    invalid = []
    for s in SMILES:
        fps = smiles_to_dense_fps(s)
        if fps is None:
            invalid.append(s)
        else:
            molecules.append(fps)
            valid_smiles.append(s)

    if invalid:
        print(f"\nWarning: {len(invalid)} invalid SMILES (skipped):")
        for s in invalid:
            print(f"  {s!r}")

    if not molecules:
        print("No valid SMILES — nothing to predict.")
        return

    # 2) Inference via the repo's predict() (downloads HF model on first call, cached after)
    print(f"\nRunning inference on {len(molecules)} valid molecule(s)...")
    probs = predict(
        molecules,
        model_path=MODEL,
        from_hub=True,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    # 3) Print results
    print("\n--- Predictions ---")
    print(f"{'P(active)':>10}    SMILES")
    print(f"{'-' * 10}    {'-' * 50}")
    for smi, p in zip(valid_smiles, probs):
        print(f"{p:>10.4f}    {smi}")

    # 4) Save to CSV
    if OUTPUT_CSV:
        df = pd.DataFrame({"SMILES": valid_smiles, "prob_active": probs})
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(df)} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
