"""
Inferència amb el model Multi: prediu M bits del fingerprint Morgan simultàniament
a partir de seqüències de tokens m/z, usant un checkpoint entrenat.

Usa els thresholds calibrats per bit desats al checkpoint (per_bit_sigmoid_threshold)
per decidir si cada bit és actiu (1) o inactiu (0). Si no existeixen, usa 0.5 per tots.
"""
import argparse
import sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "model"))

from model import TokenSequenceClassifierMulti


def main():
    parser = argparse.ArgumentParser(description='Predict multi-labels for token sequences')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_multi_latest.pth', help='Path to checkpoint file (defaults to multi-label latest checkpoint)')
    parser.add_argument('--K', type=int, required=True, help='Vocabulary size')
    parser.add_argument('--dpeak', type=int, default=16, help='Embedding dimension')
    parser.add_argument('--L', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--M', type=int, required=True, help='Number of binary labels (fixed, must match training)')
    parser.add_argument('--num-attention-layers', type=int, default=2, help='Number of attention layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--mlp-hidden-dims', type=int, nargs='+', default=[32, 16], help='MLP hidden dimensions')
    
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
    model = TokenSequenceClassifierMulti(
        K=args.K,
        dpeak=args.dpeak,
        L=args.L,
        M=args.M,
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
    if 'M' in checkpoint:
        print(f'Checkpoint M: {checkpoint["M"]}')

    # Usar els thresholds calibrats per bit desats al checkpoint; si no existeixen, usar 0.5
    per_bit_threshold = checkpoint.get('per_bit_sigmoid_threshold', None)
    if per_bit_threshold is not None:
        print(f'Usant threshold calibrat per bit del checkpoint.')
    else:
        print(f'Threshold calibrat no trobat al checkpoint; usant 0.5 per tots els bits.')
    print()

    # Predict
    model.predict(
        input_data=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=device,
        per_bit_threshold=per_bit_threshold,
    )
    
    print('\nDone!')


if __name__ == '__main__':
    main()
