"""
Entrenament del model Multi: un únic model que prediu M bits del fingerprint
Morgan simultàniament a partir de seqüències de tokens m/z.

Per defecte s'entrena sobre el subconjunt de 45 bits definit a DEFAULT_LABEL_INDICES.
Es pot especificar un subconjunt diferent amb --label-indices o --use-default-bit-subset,
o entrenar sobre tots els bits del fingerprint amb --M.

Els checkpoints i la taula de calibratge per bit es desen a --output-dir.
"""
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "model"))

from model import TokenSequenceClassifierMulti

# Subconjunt per defecte (índexs 0-based al fingerprint de 2048 bits del fitxer)
DEFAULT_LABEL_INDICES = (
    "80,314,378,561,650,656,674,728,790,807,841,875,926,1019,1057,1060,1088,"
    "1274,1380,1536,1631,1683,1750,1873,1917,157,242,419,533,701,894,912,1035,"
    "1150,1222,1341,1409,1588,1660,1721,1845,1903,1999,2012,2041"
)


def parse_label_indices(s: str) -> list[int]:
    """Converteix una cadena de índexs separats per comes en una llista d'enters validada."""
    bits = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(bits) != len(set(bits)):
        print("Error: hi ha índexs duplicats a --label-indices", file=sys.stderr)
        sys.exit(1)
    return bits


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-label token sequence classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--K", type=int, required=True, help="Vocabulary size (number of unique tokens)")
    parser.add_argument("--dpeak", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--L", type=int, required=True, help="Maximum sequence length")
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="Number of binary outputs. Si no uses --label-indices, és obligatori.",
    )
    parser.add_argument(
        "--label-indices",
        type=str,
        default=None,
        help=(
            "Índexs 0-based separats per comes: es llegeix el fingerprint complet del fitxer "
            "(veure --full-label-dim) i només s'entrenen aquests bits. "
            f"Exemple (el teu conjunt): {DEFAULT_LABEL_INDICES[:60]}..."
        ),
    )
    parser.add_argument(
        "--use-default-bit-subset",
        action="store_true",
        help=f"Equivalent a --label-indices amb el conjunt de 45 bits predefinit al script.",
    )
    parser.add_argument(
        "--full-label-dim",
        type=int,
        default=2048,
        help="Nombre d'etiquetes per línia al .txt quan s'utilitza --label-indices (Morgan 2048).",
    )
    parser.add_argument("--num-attention-layers", type=int, default=2, help="Number of attention layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="+",
        default=[32, 16],
        help="MLP hidden dimensions",
    )

    parser.add_argument("--train-file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--val-file", type=str, required=True, help="Path to validation data file")
    parser.add_argument("--test-file", type=str, default=None, help="Path to test data file (optional)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers")

    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory for tensorboard logs")
    parser.add_argument("--save-best", action="store_true", help="Save best model based on validation loss")
    parser.add_argument(
        "--figures-train-dir",
        type=str,
        default="figures/figures_training",
        help="Carpeta on desar training_summary.png al final de l'entrenament",
    )
    parser.add_argument(
        "--calibration-lambda",
        type=float,
        default=0.25,
        help=(
            "Pes de la pèrdua de calibratge marginal (mitjana σ vs freq. y=1); "
            "0.0 només desa τ sense pèrdua extra"
        ),
    )

    args = parser.parse_args()

    label_indices = None
    if args.use_default_bit_subset:
        if args.label_indices is not None:
            print("Error: no combinis --use-default-bit-subset amb --label-indices", file=sys.stderr)
            sys.exit(1)
        label_indices = parse_label_indices(DEFAULT_LABEL_INDICES)
    elif args.label_indices is not None:
        label_indices = parse_label_indices(args.label_indices)

    if label_indices is not None:
        for i in label_indices:
            if i < 0 or i >= args.full_label_dim:
                print(
                    f"Error: índex {i} fora de rang [0, {args.full_label_dim - 1}]",
                    file=sys.stderr,
                )
                sys.exit(1)
        M = len(label_indices)
        if args.M is not None and args.M != M:
            print(
                f"Avís: --M={args.M} s'ignora; amb aquest subconjunt M={M}.",
                file=sys.stderr,
            )
    else:
        if args.M is None:
            print("Error: cal --M o bé --label-indices / --use-default-bit-subset", file=sys.stderr)
            sys.exit(1)
        M = args.M

    print("Creating model...")
    model = TokenSequenceClassifierMulti(
        K=args.K,
        dpeak=args.dpeak,
        L=args.L,
        M=M,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        mlp_hidden_dims=args.mlp_hidden_dims,
    )

    model.train(
        train_data=args.train_file,
        val_data=args.val_file,
        test_data=args.test_file,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        save_best=args.save_best,
        label_indices=label_indices,
        full_label_dim=args.full_label_dim,
        figures_train_dir=args.figures_train_dir,
        calibration_lambda=args.calibration_lambda,
    )


if __name__ == "__main__":
    main()
