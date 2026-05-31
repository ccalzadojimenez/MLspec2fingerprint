"""
Entrena 45 models single (un per bit) i desa 45 gràfics de resum (un per bit).

Ara suporta entrenament en paral·lel amb CPU (ProcessPoolExecutor),
útil per màquines multi-nucli.
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent  # spec2finger/
sys.path.insert(0, str(_ROOT / "model"))
sys.path.insert(0, str(_ROOT / "training" / "multi"))

from model import TokenSequenceClassifier
from train_multi import DEFAULT_LABEL_INDICES, parse_label_indices
from train_single_bit_from_multilabel import (
    SingleBitFromFullFingerprintDataset,
    _inject_bit_index_into_checkpoint,
    _make_loader,
)


def _plot_bit_summary(
    out_path: Path,
    bit_index: int,
    split_name: str,
    accuracy: float,
    precision: float,
    recall: float,
    threshold: float,
) -> None:
    """Genera i desa un gràfic de barres amb accuracy, precision i recall
    per a un bit, incloent una línia horitzontal amb el threshold calibrat."""
    import matplotlib.pyplot as plt

    metric_names = ["Accuracy", "Precision", "Recall"]
    metric_vals = [accuracy, precision, recall]
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    bars = ax.bar(metric_names, metric_vals, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Valor")
    ax.set_title(f"Bit {bit_index} ({split_name})")
    ax.grid(axis="y", alpha=0.25)

    thr_plot = max(0.0, min(1.0, float(threshold)))
    ax.axhline(
        thr_plot,
        color="#b07aa1",
        linestyle="--",
        linewidth=2.0,
        label=f"Threshold={threshold:.4f}",
    )
    ax.legend(loc="lower right", fontsize=9)

    for b, v in zip(bars, metric_vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _metric_from_checkpoint(ckpt: Dict, key: str, fallback: float = 0.0) -> float:
    """Extreu una mètrica numèrica del checkpoint; retorna fallback si no existeix o no és convertible."""
    v = ckpt.get(key, fallback)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(fallback)


def _choose_eval_split(ckpt: Dict, use_test_if_available: bool) -> str:
    """Retorna 'test' si està disponible i es demana; altrament 'val'."""
    if use_test_if_available and ("test_acc" in ckpt):
        return "test"
    return "val"


def _train_single_bit_worker(
    cfg: Dict,
    bit: int,
    idx: int,
    total: int,
) -> Dict:
    """Funció worker executada per cada procés de ProcessPoolExecutor.
    Entrena un model Single per al bit indicat i retorna les mètriques finals."""
    # Evita oversubscription CPU quan hi ha diversos processos.
    threads = int(cfg.get("torch_threads_per_worker", 1))
    if threads > 0:
        torch.set_num_threads(threads)

    output_root = Path(cfg["output_root"])
    bit_dir = output_root / f"bit_{bit}"
    ckpt_dir = bit_dir / "checkpoints"
    log_dir = bit_dir / "logs"
    fig_train_dir = bit_dir / "figures_train"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_train_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{idx:02d}/{total}] Entrenant bit {bit} ...")

    train_ds = SingleBitFromFullFingerprintDataset(
        cfg["train_file"], bit, cfg["L"], cfg["full_label_dim"]
    )
    val_ds = SingleBitFromFullFingerprintDataset(
        cfg["val_file"], bit, cfg["L"], cfg["full_label_dim"]
    )
    test_ds: Optional[SingleBitFromFullFingerprintDataset] = None
    if cfg["test_file"]:
        test_ds = SingleBitFromFullFingerprintDataset(
            cfg["test_file"], bit, cfg["L"], cfg["full_label_dim"]
        )

    train_loader = _make_loader(train_ds, cfg["batch_size"], True, cfg["num_workers"])
    val_loader = _make_loader(val_ds, cfg["batch_size"], False, cfg["num_workers"])
    test_loader = (
        _make_loader(test_ds, cfg["batch_size"], False, cfg["num_workers"]) if test_ds else None
    )

    model = TokenSequenceClassifier(
        K=cfg["K"],
        dpeak=cfg["dpeak"],
        L=cfg["L"],
        num_attention_layers=cfg["num_attention_layers"],
        num_heads=cfg["num_heads"],
        mlp_hidden_dims=cfg["mlp_hidden_dims"],
    )

    model.train(
        train_data=train_loader,
        val_data=val_loader,
        test_data=test_loader,
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        num_workers=cfg["num_workers"],
        output_dir=str(ckpt_dir),
        log_dir=str(log_dir),
        save_best=cfg["save_best"],
        figures_train_dir=str(fig_train_dir),
    )

    latest_path = ckpt_dir / "checkpoint_multi_latest.pth"
    best_path = ckpt_dir / "checkpoint_multi_best.pth"
    _inject_bit_index_into_checkpoint(str(latest_path), bit)
    _inject_bit_index_into_checkpoint(str(best_path), bit)

    ckpt_path = best_path if (cfg["save_best"] and best_path.is_file()) else latest_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No s'ha trobat checkpoint per bit {bit}: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    split = _choose_eval_split(ckpt, bool(cfg["use_test_for_summary"]))
    acc = _metric_from_checkpoint(ckpt, f"{split}_acc", fallback=0.0)
    prec = _metric_from_checkpoint(ckpt, f"{split}_precision", fallback=0.0)
    rec = _metric_from_checkpoint(ckpt, f"{split}_recall", fallback=0.0)
    thr = _metric_from_checkpoint(ckpt, "sigmoid_threshold", fallback=0.5)

    print(
        f"Bit {bit}: {split} acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, threshold={thr:.4f}"
    )

    return {
        "bit": bit,
        "split": split,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "threshold": thr,
        "ckpt_path": str(ckpt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena 45 bits amb model single i desa 45 gràfics de mètriques+threshold "
            "(amb opció de paral·lelitzar)."
        )
    )
    parser.add_argument("--K", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--L", type=int, required=True, help="Max sequence length")
    parser.add_argument("--dpeak", type=int, default=16)
    parser.add_argument("--num-attention-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--full-label-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument(
        "--use-test-for-summary",
        action="store_true",
        help="Si hi ha test, usar mètriques de test als gràfics resum (si no, validació).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="single_45_runs",
        help="Arrel on guardar checkpoints/logs/figures per bit i els 45 resums finals.",
    )
    parser.add_argument(
        "--num-parallel-workers",
        type=int,
        default=8,
        help="Nombre de processos en paral·lel (recomanat: nombre de nuclis CPU).",
    )
    parser.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=1,
        help="Threads CPU de PyTorch per procés worker (1 recomanat en paral·lel).",
    )
    args = parser.parse_args()

    bits_45 = parse_label_indices(DEFAULT_LABEL_INDICES)
    output_root = Path(args.output_root)
    summary_dir = output_root / "bit_summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "K": args.K,
        "L": args.L,
        "dpeak": args.dpeak,
        "num_attention_layers": args.num_attention_layers,
        "num_heads": args.num_heads,
        "mlp_hidden_dims": list(args.mlp_hidden_dims),
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "full_label_dim": args.full_label_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "save_best": args.save_best,
        "use_test_for_summary": args.use_test_for_summary,
        "output_root": str(output_root),
        "torch_threads_per_worker": args.torch_threads_per_worker,
    }

    max_workers = max(1, min(args.num_parallel_workers, len(bits_45)))
    print(f"Entrenament de 45 bits amb {max_workers} worker(s) en paral·lel.")

    results: List[Dict] = []
    if max_workers == 1:
        for idx, bit in enumerate(bits_45, start=1):
            results.append(_train_single_bit_worker(cfg, bit, idx, len(bits_45)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_to_bit = {
                ex.submit(_train_single_bit_worker, cfg, bit, idx, len(bits_45)): bit
                for idx, bit in enumerate(bits_45, start=1)
            }
            for fut in as_completed(fut_to_bit):
                bit = fut_to_bit[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    raise RuntimeError(f"Ha fallat l'entrenament del bit {bit}: {e}") from e

    # Ordenem per bit per tenir sortida consistent.
    results = sorted(results, key=lambda r: r["bit"])

    csv_path = output_root / "bit_summary_metrics.csv"
    csv_rows: List[List[str]] = []
    for r in results:
        plot_path = summary_dir / f"bit_{r['bit']}_summary.png"
        _plot_bit_summary(
            plot_path,
            r["bit"],
            r["split"],
            r["acc"],
            r["prec"],
            r["rec"],
            r["threshold"],
        )
        csv_rows.append(
            [
                str(r["bit"]),
                r["split"],
                f"{r['acc']:.6f}",
                f"{r['prec']:.6f}",
                f"{r['rec']:.6f}",
                f"{r['threshold']:.6f}",
                r["ckpt_path"],
            ]
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bit_index",
                "summary_split",
                "accuracy",
                "precision",
                "recall",
                "threshold",
                "checkpoint_path",
            ]
        )
        w.writerows(csv_rows)

    print("\nFet.")
    print(f"45 gràfics desats a: {summary_dir.resolve()}")
    print(f"Resum CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
