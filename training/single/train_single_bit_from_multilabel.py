"""
Entrenament del model Single: un model independent per a un únic bit del fingerprint
Morgan a partir de seqüències de tokens m/z.

L'entrada és el mateix fitxer multi-etiqueta:
  bit0,bit1,...,bit2047;token1,token2,...
però aquest script extreu només el bit indicat per --bit-index.

Per entrenar els 45 bits en paral·lel, usar train_all_45_bits_single.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model"))

from model import TokenSequenceClassifier


class SingleBitFromFullFingerprintDataset(Dataset):
    """
    Dataset que llegeix un fitxer multi-etiqueta (2048 bits) i extreu
    únicament el bit indicat per bit_index com a etiqueta binària.
    """

    def __init__(
        self,
        file_path: str,
        bit_index: int,
        max_length: int,
        full_label_dim: int = 2048,
        pad_token: int = 0,
    ) -> None:
        self.bit_index = bit_index
        self.max_length = max_length
        self.pad_token = pad_token
        self.labels: List[int] = []
        self.sequences: List[List[int]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Línia {line_num}: format invàlid. Esperat 'labels;tokens'."
                    )
                labels_str, token_str = parts
                labels = [int(x.strip()) for x in labels_str.split(",") if x.strip()]
                if len(labels) != full_label_dim:
                    raise ValueError(
                        f"Línia {line_num}: esperades {full_label_dim} etiquetes, "
                        f"però n'hi ha {len(labels)}."
                    )
                if bit_index < 0 or bit_index >= len(labels):
                    raise ValueError(
                        f"bit_index={bit_index} fora de rang [0, {len(labels)-1}]"
                    )
                label = labels[bit_index]
                if label not in (0, 1):
                    raise ValueError(f"Línia {line_num}: label no binària al bit {bit_index}.")

                if token_str:
                    tokens = [int(t.strip()) for t in token_str.split(",") if t.strip()]
                else:
                    tokens = []

                self.labels.append(label)
                self.sequences.append(tokens)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        tokens = self.sequences[idx]
        label = float(self.labels[idx])

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        padded_tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))
        attention_mask = [1.0] * len(tokens) + [0.0] * (self.max_length - len(tokens))

        return {
            "token_indices": torch.tensor(padded_tokens, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def _inject_bit_index_into_checkpoint(path: str, bit_index: int) -> None:
    """Afegeix el camp fingerprint_bit_index al checkpoint perquè els scripts
    d'avaluació sàpiguen quin bit prediu cada model Single."""
    if not os.path.isfile(path):
        return
    ckpt = torch.load(path, map_location="cpu")
    ckpt["fingerprint_bit_index"] = int(bit_index)
    torch.save(ckpt, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena model simple per un únic bit d'un fitxer multi-etiqueta."
    )
    parser.add_argument("--bit-index", type=int, required=True, help="Índex 0-based del bit.")
    parser.add_argument("--full-label-dim", type=int, default=2048)
    parser.add_argument("--K", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--dpeak", type=int, default=16)
    parser.add_argument("--L", type=int, required=True, help="Longitud màxima de seqüència.")
    parser.add_argument("--num-attention-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./checkpoints_single_bit")
    parser.add_argument("--log-dir", type=str, default="./logs_single_bit")
    parser.add_argument("--figures-train-dir", type=str, default="figures/figures_training")
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    train_ds = SingleBitFromFullFingerprintDataset(
        args.train_file, args.bit_index, args.L, args.full_label_dim
    )
    val_ds = SingleBitFromFullFingerprintDataset(
        args.val_file, args.bit_index, args.L, args.full_label_dim
    )
    test_ds: Optional[SingleBitFromFullFingerprintDataset]
    test_ds = None
    if args.test_file:
        test_ds = SingleBitFromFullFingerprintDataset(
            args.test_file, args.bit_index, args.L, args.full_label_dim
        )

    train_loader = _make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = _make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = (
        _make_loader(test_ds, args.batch_size, False, args.num_workers) if test_ds else None
    )

    model = TokenSequenceClassifier(
        K=args.K,
        dpeak=args.dpeak,
        L=args.L,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        mlp_hidden_dims=args.mlp_hidden_dims,
    )

    print(f"Entrenant model simple per bit {args.bit_index} ...")
    model.train(
        train_data=train_loader,
        val_data=val_loader,
        test_data=test_loader,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        save_best=args.save_best,
        figures_train_dir=args.figures_train_dir,
    )

    _inject_bit_index_into_checkpoint(
        os.path.join(args.output_dir, "checkpoint_multi_latest.pth"), args.bit_index
    )
    _inject_bit_index_into_checkpoint(
        os.path.join(args.output_dir, "checkpoint_multi_best.pth"), args.bit_index
    )
    print("Fet.")


if __name__ == "__main__":
    main()
