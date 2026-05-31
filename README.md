# spec2finger

Deep learning model that predicts Morgan fingerprint (ECFP4) bits from mass spectrometry (MZ) token sequences, using a permutation-invariant Transformer architecture.

## Overview

Given a mass spectrum (a set of MZ peaks), the model predicts which bits of the Morgan fingerprint (2048 bits, radius=2) are active for the corresponding molecule. Two approaches are implemented and compared:

- **Single method**: 45 independent models, one per fingerprint bit.
- **Multi method**: one single model with 45 simultaneous outputs.

Both approaches use the same underlying architecture: token embeddings + permutation-invariant self-attention + attention pooling + MLP classifier.

## Architecture

MZ values are tokenized as `int(mz * 100)` (e.g., 10.145 → 1014). The model treats each spectrum as an **unordered set** of tokens — no positional encoding is used.

**TokenSequenceClassifier** (Single method, one bit):
1. Token embedding lookup table
2. Shared self-attention stack (multi-head, no positional encoding)
3. One attention pooling query → fixed-size vector
4. MLP → single binary logit

**TokenSequenceClassifierMulti** (Multi method, M bits):
Same backbone, but with M independent pooling queries and M independent MLPs — one per bit.

## Project structure

All scripts are run from the `spec2finger/` directory.

```
spec2finger/
├── model/
│   ├── model.py              # Model architectures + training/prediction logic
│   └── data_loader.py        # PyTorch Dataset and DataLoader utilities
│
├── preprocessing/
│   ├── data_preprocess.py    # CSV (MZ + SMILES) → train/val .txt files
│   └── prepare_test_data.py  # Same conversion for the test split
│
├── training/
│   ├── multi/
│   │   └── train_multi.py               # Train one model predicting 45 bits at once
│   └── single/
│       ├── train_all_45_bits_single.py  # Train 45 independent models (parallelised)
│       └── train_single_bit_from_multilabel.py  # Train one model for one specific bit
│
├── postprocessing/
│   ├── evaluate_multi_45bits_on_test.py   # Evaluate Multi model on test set
│   ├── evaluate_single_45bits_on_test.py  # Evaluate 45 Single models on test set
│   ├── predict.py                         # Run inference with a Single checkpoint
│   └── predict_multi.py                   # Run inference with the Multi checkpoint
│
├── data/                    # Train / val / test .txt files and original CSVs (not included in repo)
├── requirements.txt
└── README.md
```

## Data format

Each line in the `.txt` data files:
```
bit0,bit1,...,bit2047;token1,token2,...,tokenN
```
- **Labels** (left of `;`): 2048 binary values, the Morgan fingerprint bits.
- **Tokens** (right of `;`): MZ peaks encoded as `int(mz * 100)`, deduplicated and sorted.

Example: `0,1,0,...,1;1014,2345,789`

## Installation

```bash
pip install -r requirements.txt
```

RDKit is only required for preprocessing:
```bash
pip install rdkit
# If that fails: conda install -c conda-forge rdkit
```

## Typical workflow

### 1. Preprocess raw data

```bash
# Generate train and val files
python preprocessing/data_preprocess.py

# Generate test file
python preprocessing/prepare_test_data.py \
    --input-csv data/merge_final_test\(in\).csv \
    --output-txt data/fingerMorgan_mz_test.txt
```

### 2. Train

**Multi method** (one model, 45 bits simultaneously):
```bash
python training/multi/train_multi.py \
    --K <vocab_size> --L <max_seq_len> \
    --use-default-bit-subset \
    --train-file data/fingerMorgan_mz_train.txt \
    --val-file data/fingerMorgan_mz_val.txt \
    --test-file data/fingerMorgan_mz_test.txt \
    --epochs 10 --save-best
```

**Single method** (45 independent models, parallelised):
```bash
python training/single/train_all_45_bits_single.py \
    --K <vocab_size> --L <max_seq_len> \
    --train-file data/fingerMorgan_mz_train.txt \
    --val-file data/fingerMorgan_mz_val.txt \
    --test-file data/fingerMorgan_mz_test.txt \
    --output-root single_45_runs \
    --epochs 10 --save-best --num-parallel-workers 8
```

### 3. Evaluate on test set

```bash
# Multi model
python postprocessing/evaluate_multi_45bits_on_test.py \
    --checkpoint checkpoints/checkpoint_best.pth \
    --test-file data/fingerMorgan_mz_test.txt

# Single models
python postprocessing/evaluate_single_45bits_on_test.py \
    --models-root single_45_runs \
    --test-file data/fingerMorgan_mz_test.txt
```

### 4. Run inference

```bash
# Multi model
python postprocessing/predict_multi.py \
    --checkpoint checkpoints/checkpoint_best.pth

# Single model (one bit)
python postprocessing/predict.py \
    --checkpoint single_45_runs/bit_80/checkpoints/checkpoint_multi_best.pth
```

## Customising the bit subset

By default, the model trains on a predefined subset of 45 bits (`DEFAULT_LABEL_INDICES` in `training/multi/train_multi.py`). You can change this in three ways:

**1. Use the default 45 bits:**
```bash
python training/multi/train_multi.py --use-default-bit-subset ...
```

**2. Specify your own set of bits (any number):**
```bash
python training/multi/train_multi.py --label-indices "80,314,650,1019" ...
```
Pass any comma-separated list of bit indices (0-based, between 0 and 2047). You can use as many or as few bits as you want.

**3. Change the default subset permanently:**
Edit the `DEFAULT_LABEL_INDICES` variable at the top of `training/multi/train_multi.py`:
```python
DEFAULT_LABEL_INDICES = "80,314,650,1019,..."
```

> **Note:** `--use-default-bit-subset` and `--label-indices` cannot be used together.

## Key implementation notes

- `model.py` defines a custom `.train()` method that overrides `nn.Module.train()`. To set eval mode inside the codebase, use `nn.Module.train(model, False)` — never `model.eval()`.
- Checkpoints include a per-bit calibrated threshold (`per_bit_sigmoid_threshold`) computed after training so that predicted positive rates match the training set frequencies.
- The default 45-bit subset is defined in `training/multi/train_multi.py` as `DEFAULT_LABEL_INDICES`.
