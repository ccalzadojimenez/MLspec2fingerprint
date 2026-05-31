"""
Inferència amb el model Single: prediu un únic bit del fingerprint Morgan
a partir de seqüències de tokens m/z, usant un checkpoint entrenat.

Usa el threshold calibrat desat al checkpoint (sigmoid_threshold) per
decidir si el bit és actiu (1) o inactiu (0). Si no existeix, usa 0.5.
"""
import argparse
import sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "model"))  # per trobar model.py i data_loader.py

from model import TokenSequenceClassifier


def main():
    parser = argparse.ArgumentParser(description='Predict labels for token sequences')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--K', type=int, required=True, help='Vocabulary size')
    parser.add_argument('--dpeak', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--L', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--num-attention-layers', type=int, default=2, help='Number of attention layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp-hidden-dims', type=int, nargs='+', default=[256, 128], help='MLP hidden dimensions')
    
    # Input/Output
    parser.add_argument('--input-file', type=str, required=True, help='Input file with token sequences')
    parser.add_argument('--output-file', type=str, required=True, help='Output file for predictions')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    print('Creating model...')
    model = TokenSequenceClassifier(
        K=args.K,
        dpeak=args.dpeak,
        L=args.L,
        num_attention_layers=args.num_attention_layers,
        num_heads=args.num_heads,
        mlp_hidden_dims=args.mlp_hidden_dims
    )
    
    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f'Checkpoint epoch: {checkpoint["epoch"]}')
    if 'val_acc' in checkpoint:
        print(f'Validation accuracy: {checkpoint["val_acc"]:.4f}')
    if 'test_acc' in checkpoint:
        print(f'Test accuracy: {checkpoint["test_acc"]:.4f}')

    # Usar el threshold calibrat desat al checkpoint; si no existeix, usar 0.5
    threshold = float(checkpoint.get('sigmoid_threshold', 0.5))
    print(f'Threshold: {threshold:.4f}')
    print()

    # Predict
    model.predict(
        input_data=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=device,
        threshold=threshold,
    )
    
    print('\nDone!')


if __name__ == '__main__':
    main()
