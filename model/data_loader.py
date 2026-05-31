"""
Data loading utilities for token sequence classification.
"""
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TokenSequenceDataset(Dataset):
    """
    Dataset for token sequences with binary labels.

    Expected format: Each line is "bit0,bit1,...,bit2047;token1,token2,token3,..."
    where labels are 2048 binary values and tokens are comma-separated integers.
    """

    def __init__(self, file_path, max_length, pad_token=0):
        """
        Initialize the dataset.

        Args:
            file_path: Path to the data file
            max_length: Maximum sequence length (L)
            pad_token: Token ID to use for padding (default: 0)
        """
        self.max_length = max_length
        self.pad_token = pad_token

        self.labels = []
        self.sequences = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(';', 1)
                if len(parts) != 2:
                    continue

                labels_str = parts[0]
                if ',' in labels_str:
                    label = [int(x.strip()) for x in labels_str.split(',') if x.strip()]
                else:
                    label = [int(labels_str)]

                token_str = parts[1]
                if token_str:
                    tokens = [int(t.strip()) for t in token_str.split(',') if t.strip()]
                else:
                    tokens = []

                self.labels.append(label)
                self.sequences.append(tokens)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            token_indices: Padded token sequence (length max_length)
            label: Binary label tensor with all fingerprint bits
            attention_mask: Mask indicating valid tokens (1) vs padding (0)
        """
        tokens = self.sequences[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create padded sequence
        padded_tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
        
        return {
            'token_indices': torch.tensor(padded_tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }


def create_data_loaders(
    train_file,
    val_file,
    test_file=None,
    max_length=100,
    batch_size=128,
    num_workers=0,
    pad_token=0
):
    """
    Create data loaders for training, validation, and optionally test sets.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        test_file: Optional path to test data file
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        pad_token: Token ID to use for padding
    
    Returns:
        train_loader, val_loader, (test_loader if test_file provided)
    """
    train_dataset = TokenSequenceDataset(train_file, max_length, pad_token)
    val_dataset = TokenSequenceDataset(val_file, max_length, pad_token)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    if test_file is not None:
        test_dataset = TokenSequenceDataset(test_file, max_length, pad_token)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


class TokenSequenceMultiLabelDataset(Dataset):
    """
    Dataset for token sequences with multi-label binary outputs.
    
    Expected format: Each line is "label1,label2,...,labelM;token1,token2,token3,..."
    where each label is 0 or 1, and tokens are comma-separated integers.
    M is fixed and should be consistent across all samples.
    """
    
    def __init__(
        self,
        file_path,
        max_length,
        M,
        pad_token=0,
        label_indices: Optional[Sequence[int]] = None,
        full_label_dim: int = 2048,
    ):
        """
        Initialize the multi-label dataset.
        
        Args:
            file_path: Path to the data file
            max_length: Maximum sequence length (L)
            M: Number of binary labels per sample after loading (len(label_indices) if subset)
            pad_token: Token ID to use for padding (default: 0)
            label_indices: Si no és None, el fitxer té `full_label_dim` etiquetes per línia
                (p. ex. 2048 bits Morgan) i només es carreguen les posicions indicades (índexs 0-based).
            full_label_dim: Nombre d'etiquetes al fitxer quan s'utilitza label_indices.
        """
        self.max_length = max_length
        self.M = M
        self.pad_token = pad_token
        self.full_label_dim = full_label_dim
        if label_indices is not None:
            self.label_indices = [int(i) for i in label_indices]
            if len(self.label_indices) != M:
                raise ValueError(
                    f"len(label_indices) ({len(self.label_indices)}) ha de coincidir amb M ({M})"
                )
            if any(i < 0 or i >= full_label_dim for i in self.label_indices):
                raise ValueError(
                    f"Tots els índexs han d'estar en [0, {full_label_dim - 1}]. "
                    f"Rebut: min={min(self.label_indices)}, max={max(self.label_indices)}"
                )
        else:
            self.label_indices = None
        
        self.labels = []
        self.sequences = []
        
        # Load and parse the file
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split labels and tokens
                parts = line.split(';', 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid format at line {line_num}: expected 'label1,label2,...,labelM;token1,token2,...'")
                
                labels_str = parts[0]
                token_str = parts[1]
                
                # Parse labels
                if labels_str:
                    labels = [int(l.strip()) for l in labels_str.split(',') if l.strip()]
                else:
                    labels = []
                
                if self.label_indices is not None:
                    if len(labels) != full_label_dim:
                        raise ValueError(
                            f"Línia {line_num}: amb --label-indices calen {full_label_dim} "
                            f"etiquetes al fitxer, n'hi ha {len(labels)}."
                        )
                    labels = [labels[i] for i in self.label_indices]
                else:
                    if len(labels) != M:
                        raise ValueError(
                            f"Invalid number of labels at line {line_num}: "
                            f"expected {M} labels, got {len(labels)}. "
                            f"Labels: {labels}"
                        )
                
                # Validate labels are binary
                for i, label in enumerate(labels):
                    if label not in [0, 1]:
                        raise ValueError(
                            f"Invalid label at line {line_num}, position {i}: "
                            f"expected 0 or 1, got {label}"
                        )
                
                # Parse tokens
                if token_str:
                    tokens = [int(t.strip()) for t in token_str.split(',') if t.strip()]
                else:
                    tokens = []
                
                self.labels.append(labels)
                self.sequences.append(tokens)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            token_indices: Padded token sequence (length max_length)
            labels: Multi-label binary tensor (length M)
            attention_mask: Mask indicating valid tokens (1) vs padding (0)
        """
        tokens = self.sequences[idx]
        labels = self.labels[idx]
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create padded sequence
        padded_tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
        
        return {
            'token_indices': torch.tensor(padded_tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float),  # Shape: (M,)
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }


def create_data_loaders_multi(
    train_file,
    val_file,
    test_file=None,
    max_length=100,
    M=3,
    batch_size=128,
    num_workers=0,
    pad_token=0,
    label_indices: Optional[Sequence[int]] = None,
    full_label_dim: int = 2048,
):
    """
    Create data loaders for multi-label training, validation, and optionally test sets.
    
    Args:
        train_file: Path to training data file
        val_file: Path to validation data file
        test_file: Optional path to test data file
        max_length: Maximum sequence length
        M: Number of binary labels per batch (després del subset, ha de ser len(label_indices))
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        pad_token: Token ID to use for padding
        label_indices: Opcional: índexs 0-based al fingerprint complet del fitxer
        full_label_dim: Etiquetes per línia al fitxer si s'usa label_indices (per defecte 2048)
    
    Returns:
        train_loader, val_loader, (test_loader if test_file provided)
    """
    ds_kw = dict(
        max_length=max_length,
        M=M,
        pad_token=pad_token,
        label_indices=label_indices,
        full_label_dim=full_label_dim,
    )
    train_dataset = TokenSequenceMultiLabelDataset(train_file, **ds_kw)
    val_dataset = TokenSequenceMultiLabelDataset(val_file, **ds_kw)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    if test_file is not None:
        test_dataset = TokenSequenceMultiLabelDataset(test_file, **ds_kw)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

