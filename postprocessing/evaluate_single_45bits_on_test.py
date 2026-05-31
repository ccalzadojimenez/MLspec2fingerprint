"""
Avalua els 45 models single (1 model = 1 bit) sobre dades de test i genera:
- heatmap de accuracy / precision / recall per bit
- CSV de mètriques per bit

Quan precision o recall són 0/0, es deixen com a NaN i al heatmap la casella queda en blanc.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "model"))
sys.path.insert(0, str(_ROOT / "training" / "multi"))

from model import TokenSequenceClassifier
from train_multi import DEFAULT_LABEL_INDICES, parse_label_indices


def _load_test_data_full(
    test_file: str, full_label_dim: int = 2048
) -> Tuple[np.ndarray, List[List[int]]]:
    """Llegeix fitxer labels;tokens i retorna labels (N, full_label_dim) + seqüències."""
    labels_rows: List[List[int]] = []
    sequences: List[List[int]] = []

    with open(test_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(";", 1)
            if len(parts) != 2:
                raise ValueError(f"Línia {line_num}: format invàlid (esperat labels;tokens)")
            labels_str, token_str = parts
            labels = [int(x.strip()) for x in labels_str.split(",") if x.strip()]
            if len(labels) != full_label_dim:
                raise ValueError(
                    f"Línia {line_num}: esperades {full_label_dim} etiquetes, n'hi ha {len(labels)}"
                )
            tokens = [int(t.strip()) for t in token_str.split(",") if t.strip()] if token_str else []
            labels_rows.append(labels)
            sequences.append(tokens)

    return np.asarray(labels_rows, dtype=np.int64), sequences


def _pad_batch(batch_sequences: List[List[int]], max_len: int, pad_token: int = 0):
    """Aplica padding a un batch de seqüències fins a max_len i retorna tensors de tokens i màscara."""
    batch_tokens = []
    batch_masks = []
    for tokens in batch_sequences:
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        padded = tokens + [pad_token] * (max_len - len(tokens))
        mask = [1.0] * len(tokens) + [0.0] * (max_len - len(tokens))
        batch_tokens.append(padded)
        batch_masks.append(mask)
    return (
        torch.tensor(batch_tokens, dtype=torch.long),
        torch.tensor(batch_masks, dtype=torch.float32),
    )


def _find_checkpoint_for_bit(models_root: Path, bit: int) -> Path:
    """Retorna el checkpoint del bit indicat; prefereix el best, cau al latest si no existeix."""
    ckpt_best = models_root / f"bit_{bit}" / "checkpoints" / "checkpoint_multi_best.pth"
    ckpt_latest = models_root / f"bit_{bit}" / "checkpoints" / "checkpoint_multi_latest.pth"
    if ckpt_best.is_file():
        return ckpt_best
    if ckpt_latest.is_file():
        return ckpt_latest
    raise FileNotFoundError(f"No s'ha trobat checkpoint per bit {bit} a {ckpt_best.parent}")


def _compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Retorna (tp, tn, fp, fn) per a vectors binaris aplanats."""
    yt = y_true.astype(np.int64).ravel()
    yp = y_pred.astype(np.int64).ravel()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tp, tn, fp, fn


def _safe_precision(tp: int, fp: int) -> float:
    """Precision amb protecció de divisió per zero; retorna NaN si tp+fp=0."""
    den = tp + fp
    if den == 0:
        return float("nan")
    return float(tp / den)


def _safe_recall(tp: int, fn: int) -> float:
    """Recall amb protecció de divisió per zero; retorna NaN si tp+fn=0."""
    den = tp + fn
    if den == 0:
        return float("nan")
    return float(tp / den)


def _safe_f1(precision: float, recall: float) -> float:
    """F1 amb protecció de NaN i divisió per zero."""
    den = precision + recall
    if np.isnan(precision) or np.isnan(recall) or den == 0:
        return float("nan")
    return float((2.0 * precision * recall) / den)


def _plot_heatmap(bits_sorted: np.ndarray, values_sorted: np.ndarray, out_png: Path) -> None:
    """
    values_sorted shape (n_bits, 4): [accuracy, precision, recall, f1]
    Caselles amb NaN es mostren en blanc.
    """
    import matplotlib.pyplot as plt

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    masked = np.ma.masked_invalid(values_sorted)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig_h = max(8.5, 0.23 * len(bits_sorted))
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    im = ax.imshow(masked, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Valor mètrica")

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_yticks(np.arange(len(bits_sorted)))
    ax.set_yticklabels([str(b) for b in bits_sorted], fontsize=8)
    ax.set_xlabel("Mètrica")
    ax.set_ylabel("Bit")
    ax.set_title("Models single sobre test: accuracy / precision / recall / F1 per bit")

    for i in range(values_sorted.shape[0]):
        for j in range(values_sorted.shape[1]):
            v = values_sorted[i, j]
            if np.isnan(v):
                continue  # NaN: deixem la casella en blanc
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.65 else "black",
                fontsize=7,
            )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_global_metrics_summary(
    out_png: Path, accuracy: float, precision: float, recall: float, f1: float
) -> None:
    """Histograma/barres de mètriques globals (mateix estil que al multi)."""
    import matplotlib.pyplot as plt

    names = ["Accuracy", "Precision", "Recall", "F1"]
    vals = [accuracy, precision, recall, f1]
    colors = ["#a6bddb", "#fdbb84", "#c994c7", "#80b1d3"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Valor")
    ax.set_title("Resum de mètriques globals (models single sobre test)")
    ax.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, vals):
        txt = "NA" if np.isnan(v) else f"{v:.3f}"
        y = 0.02 if np.isnan(v) else (v + 0.02)
        ax.text(b.get_x() + b.get_width() / 2, y, txt, ha="center", va="bottom")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalua 45 models single sobre test i crea heatmap per bit."
    )
    parser.add_argument("--test-file", type=str, default="data/fingerMorgan_mz_test.txt")
    parser.add_argument("--models-root", type=str, default="single_45_runs")
    parser.add_argument("--K", type=int, default=200000)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--dpeak", type=int, default=16)
    parser.add_argument("--num-attention-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--full-label-dim", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/figures_postprocessing/single_45_on_test",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bits_45 = parse_label_indices(DEFAULT_LABEL_INDICES)
    models_root = Path(args.models_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_full, sequences = _load_test_data_full(args.test_file, args.full_label_dim)
    n_samples = y_full.shape[0]
    if n_samples == 0:
        raise ValueError("No hi ha mostres al fitxer de test.")

    per_bit_rows: List[Dict] = []

    for bit in bits_45:
        ckpt_path = _find_checkpoint_for_bit(models_root, bit)
        ckpt = torch.load(str(ckpt_path), map_location=device)

        model = TokenSequenceClassifier(
            K=args.K,
            dpeak=args.dpeak,
            L=args.L,
            num_attention_layers=args.num_attention_layers,
            num_heads=args.num_heads,
            mlp_hidden_dims=args.mlp_hidden_dims,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        torch.nn.Module.train(model, False)

        threshold = float(ckpt.get("sigmoid_threshold", 0.5))

        probs_parts: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n_samples, args.batch_size):
                batch_seq = sequences[i : i + args.batch_size]
                token_indices, attention_mask = _pad_batch(batch_seq, args.L, pad_token=0)
                token_indices = token_indices.to(device)
                attention_mask = attention_mask.to(device)
                logits = model(token_indices, attention_mask).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_parts.append(probs)

        probs_all = np.concatenate(probs_parts, axis=0)
        y_true = y_full[:, bit].astype(np.int64)
        y_pred = (probs_all > threshold).astype(np.int64)
        tp, tn, fp, fn = _compute_confusion(y_true, y_pred)
        acc = float((tp + tn) / (tp + tn + fp + fn))
        prec = _safe_precision(tp, fp)  # NaN si 0/0
        rec = _safe_recall(tp, fn)      # NaN si 0/0
        f1 = _safe_f1(prec, rec)

        per_bit_rows.append(
            {
                "bit": bit,
                "checkpoint": str(ckpt_path),
                "threshold": threshold,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )

    # Ordenem per número de bit per tenir sortida consistent
    per_bit_rows = sorted(per_bit_rows, key=lambda r: r["bit"])
    bits_sorted = np.asarray([r["bit"] for r in per_bit_rows], dtype=int)
    values_sorted = np.asarray(
        [[r["accuracy"], r["precision"], r["recall"], r["f1"]] for r in per_bit_rows],
        dtype=float,
    )

    # Heatmap
    _plot_heatmap(bits_sorted, values_sorted, output_dir / "per_bit_metrics_heatmap_test_single.png")

    # CSV (NaN en blanc)
    csv_path = output_dir / "per_bit_metrics_test_single.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bit",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "tp",
                "tn",
                "fp",
                "fn",
                "checkpoint",
            ]
        )
        for r in per_bit_rows:
            w.writerow(
                [
                    r["bit"],
                    f"{r['threshold']:.6f}",
                    f"{r['accuracy']:.6f}",
                    "" if np.isnan(r["precision"]) else f"{r['precision']:.6f}",
                    "" if np.isnan(r["recall"]) else f"{r['recall']:.6f}",
                    "" if np.isnan(r["f1"]) else f"{r['f1']:.6f}",
                    r["tp"],
                    r["tn"],
                    r["fp"],
                    r["fn"],
                    r["checkpoint"],
                ]
            )

    # Resum global micro
    tp_g = int(sum(r["tp"] for r in per_bit_rows))
    tn_g = int(sum(r["tn"] for r in per_bit_rows))
    fp_g = int(sum(r["fp"] for r in per_bit_rows))
    fn_g = int(sum(r["fn"] for r in per_bit_rows))
    acc_g = float((tp_g + tn_g) / (tp_g + tn_g + fp_g + fn_g))
    prec_g = _safe_precision(tp_g, fp_g)
    rec_g = _safe_recall(tp_g, fn_g)
    f1_g = _safe_f1(prec_g, rec_g)

    _plot_global_metrics_summary(
        output_dir / "summary_metrics_test_single.png",
        acc_g,
        prec_g,
        rec_g,
        f1_g,
    )

    summary_path = output_dir / "summary_test_single_45bits.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Test file: {args.test_file}\n")
        f.write(f"Models root: {models_root}\n")
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Bits: {len(bits_45)}\n\n")
        f.write("Mètriques globals (micro):\n")
        f.write(f"  Accuracy : {acc_g:.6f}\n")
        f.write(f"  Precision: {'NA' if np.isnan(prec_g) else f'{prec_g:.6f}'}\n")
        f.write(f"  Recall   : {'NA' if np.isnan(rec_g) else f'{rec_g:.6f}'}\n")
        f.write(f"  F1       : {'NA' if np.isnan(f1_g) else f'{f1_g:.6f}'}\n")

    print("Fet.")
    print(f"Heatmap: {output_dir / 'per_bit_metrics_heatmap_test_single.png'}")
    print(f"Histograma mètriques globals: {output_dir / 'summary_metrics_test_single.png'}")
    print(f"CSV: {csv_path}")
    print(f"Resum global: {summary_path}")


if __name__ == "__main__":
    main()
