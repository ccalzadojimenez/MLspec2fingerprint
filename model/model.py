"""
Model de deep learning per a la predicció de bits del fingerprint Morgan (ECFP4)
a partir de seqüències de tokens m/z.

Conté dues arquitectures basades en Transformer sense codificació posicional
(invariants a la permutació dels pics de l'espectre):

  - TokenSequenceClassifier: model Single — prediu un únic bit del fingerprint.
  - TokenSequenceClassifierMulti: model Multi — prediu M bits simultàniament
    amb un backbone compartit i M parells independents (pooling + MLP).

Ambdues classes inclouen els mètodes train() i predict() directament.

NOTA: train() sobreescriu nn.Module.train(). Per posar el model en mode
avaluació des del codi, cal usar nn.Module.train(model, False) i mai model.eval().

Funcions auxiliars incloses:
  - Mètriques: MCC (binari i multi-etiqueta), precisió, recall.
  - Calibratge: càlcul d'umbrals per bit per igualar la freqüència de predicció
    positiva a la freqüència real del conjunt d'entrenament.
  - Figures: corbes d'entrenament i taula de calibratge (requereix matplotlib).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os
import numpy as np
from typing import Union, Optional, Tuple, List

try:
    from sklearn.metrics import matthews_corrcoef
except ImportError:
    matthews_corrcoef = None


def _safe_binary_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Matthews correlation for binary vectors (flattened ok).
    Returns NaN if y_true has a single class (MCC undefined).
    """
    if matthews_corrcoef is None:
        return float("nan")
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(matthews_corrcoef(y_true, y_pred))


def _multilabel_mcc_micro_macro(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Multi-label (N, M), valors {0,1}.
    - micro: un sol MCC sobre totes les parelles (mostra, etiqueta) — comú en multi-etiqueta.
    - macro: mitjana del MCC per etiqueta (només etiquetes on y_true té les dues classes).
    """
    if matthews_corrcoef is None:
        return float("nan"), float("nan")
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.ndim != 2 or y_pred.shape != y_true.shape:
        raise ValueError("y_true i y_pred han de tenir forma (N, M) i coincidir.")

    mcc_micro = _safe_binary_mcc(y_true, y_pred)

    per_label = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        if np.unique(yt).size < 2:
            continue
        per_label.append(float(matthews_corrcoef(yt, y_pred[:, j])))
    mcc_macro = float(np.mean(per_label)) if per_label else float("nan")
    return mcc_micro, mcc_macro


def _format_mcc(x: float) -> str:
    """Escriu el MCC o 'n/a' si és NaN (MCC indefinit)."""
    if isinstance(x, float) and np.isnan(x):
        return "n/a"
    return f"{x:.4f}"


def _binary_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Precision i recall binaris amb protecció de divisió per zero."""
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def _binary_threshold_for_positive_rate(probs: np.ndarray, target_pos_rate: float) -> float:
    """
    Tria tau tal que mean(probs > tau) ≈ target_pos_rate (calibratge marginal binari).
    """
    p = float(np.clip(float(target_pos_rate), 0.0, 1.0))
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.size == 0:
        return 0.5
    if p <= 0.0:
        return float(np.max(probs) + 1e-6)
    if p >= 1.0:
        return float(np.min(probs) - 1e-6)
    return float(np.quantile(probs, 1.0 - p))


def _gather_probs_and_labels_binary(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recull probabilitats i etiquetes binàries sobre un loader."""
    nn.Module.train(model, False)
    p_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            token_indices = batch["token_indices"].to(device)
            labels = batch["label"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(token_indices, attention_mask).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            p_parts.append(probs)
            y_parts.append(labels.cpu().numpy())
    if not p_parts:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.concatenate(p_parts, axis=0), np.concatenate(y_parts, axis=0)


def _per_bit_thresholds_for_positive_rate(probs: np.ndarray, target_pos_rate: np.ndarray) -> np.ndarray:
    """
    Per cada bit j, tria tau[j] tal que mean(probs[:, j] > tau[j]) ≈ target_pos_rate[j]
    (quantils; probs N×M, target_pos_rate shape (M,)).
    """
    n, m = probs.shape
    if m != target_pos_rate.shape[0]:
        raise ValueError("target_pos_rate ha de tenir forma (M,)")
    tau = np.zeros(m, dtype=np.float64)
    for j in range(m):
        p = float(np.clip(float(target_pos_rate[j]), 0.0, 1.0))
        col = probs[:, j]
        if p <= 0.0:
            tau[j] = float(np.max(col) + 1e-6)
        elif p >= 1.0:
            tau[j] = float(np.min(col) - 1e-6)
        else:
            tau[j] = float(np.quantile(col, 1.0 - p))
    return tau


def _gather_probs_and_labels_multilabel(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    # No usar model.eval(): crida self.train(False) i aquí .train és el mètode d'entrenament
    # personalitzat (train_data, val_data, ...). Cal el train de nn.Module.
    nn.Module.train(model, False)
    p_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            token_indices = batch["token_indices"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(token_indices, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            p_parts.append(probs)
            y_parts.append(labels.cpu().numpy())
    if not p_parts:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return np.concatenate(p_parts, axis=0), np.concatenate(y_parts, axis=0)


def _save_calibration_table_multilabel(
    freq_true: np.ndarray,
    freq_pred_05: np.ndarray,
    freq_pred_cal: np.ndarray,
    mean_prob: np.ndarray,
    thresholds: np.ndarray,
    fingerprint_positions: Optional[List[int]],
    figures_dir: str,
    verbose: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    CSV amb columnes de calibratge + scatter freq. real y=1 (eix Y) vs freq. predicció 1 (eix X).
    """
    os.makedirs(figures_dir, exist_ok=True)
    m = freq_true.shape[0]
    path_csv = os.path.join(figures_dir, "calibration_per_bit.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bit_index_model",
                "fingerprint_bit_index",
                "freq_true_y1",
                "freq_pred1_threshold_0.5",
                "freq_pred1_calibrated",
                "mean_predicted_prob",
                "threshold_sigmoid_tau",
            ]
        )
        for j in range(m):
            fpos = fingerprint_positions[j] if fingerprint_positions is not None else j
            w.writerow(
                [
                    j,
                    fpos,
                    f"{freq_true[j]:.6f}",
                    f"{freq_pred_05[j]:.6f}",
                    f"{freq_pred_cal[j]:.6f}",
                    f"{mean_prob[j]:.6f}",
                    f"{thresholds[j]:.6f}",
                ]
            )
    if verbose:
        print(f"Taula de calibratge desada: {path_csv}")

    path_png: Optional[str] = None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("Avís: matplotlib no disponible; no es genera calibration_scatter.png")
        return path_csv, path_png

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = (-0.02, 1.02)
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.6, label="y = x (calibrat ideal)")
    ax.scatter(
        freq_pred_cal,
        freq_true,
        s=22,
        alpha=0.75,
        marker="o",
        label="Pred. 1 (umbral τ per bit)",
    )
    ax.scatter(
        freq_pred_05,
        freq_true,
        s=18,
        alpha=0.45,
        marker="x",
        label="Pred. 1 (umbral 0.5)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Freq. predicció = 1 (mostres train)")
    ax.set_ylabel("Freq. real y = 1 (mostres train)")
    ax.set_title("Calibratge marginal per bit (train)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_png = os.path.join(figures_dir, "calibration_scatter.png")
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"Gràfic de calibratge desat: {path_png}")
    return path_csv, path_png


def _save_training_figure_binary(
    history: dict,
    figures_dir: str,
    title: str = "Entrenament",
    verbose: bool = True,
) -> Optional[str]:
    """Gràfic loss + accuracy (train/val/test) per classificador binari."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("Avís: matplotlib no disponible; no es guarda figures_train.")
        return None

    os.makedirs(figures_dir, exist_ok=True)
    n = len(history["train_loss"])
    ep = np.arange(1, n + 1)
    has_test = bool(history.get("test_loss")) and len(history["test_loss"]) == n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax1.plot(ep, history["train_loss"], "o-", label="Train", linewidth=1.5, markersize=4)
    ax1.plot(ep, history["val_loss"], "s-", label="Validació", linewidth=1.5, markersize=4)
    if has_test:
        ax1.plot(ep, history["test_loss"], "^-", label="Test", linewidth=1.5, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, history["train_acc"], "o-", label="Train", linewidth=1.5, markersize=4)
    ax2.plot(ep, history["val_acc"], "s-", label="Validació", linewidth=1.5, markersize=4)
    if has_test:
        ax2.plot(ep, history["test_acc"], "^-", label="Test", linewidth=1.5, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.02)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(figures_dir, "training_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"Gràfic d'entrenament desat: {path}")
    return path


def _save_training_figure_multilabel(
    history: dict,
    figures_dir: str,
    title: str = "Entrenament multi-etiqueta",
    verbose: bool = True,
) -> Optional[str]:
    """Gràfic 2×2: loss, accuracy, MCC micro, MCC macro."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose:
            print("Avís: matplotlib no disponible; no es guarda figures_train.")
        return None

    os.makedirs(figures_dir, exist_ok=True)
    n = len(history["train_loss"])
    ep = np.arange(1, n + 1)
    has_test = bool(history.get("test_loss")) and len(history["test_loss"]) == n

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(ep, history["train_loss"], "o-", label="Train", linewidth=1.2, markersize=3)
    ax.plot(ep, history["val_loss"], "s-", label="Validació", linewidth=1.2, markersize=3)
    if has_test:
        ax.plot(ep, history["test_loss"], "^-", label="Test", linewidth=1.2, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, history["train_acc"], "o-", label="Train", linewidth=1.2, markersize=3)
    ax.plot(ep, history["val_acc"], "s-", label="Validació", linewidth=1.2, markersize=3)
    if has_test:
        ax.plot(ep, history["test_acc"], "^-", label="Test", linewidth=1.2, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (per etiqueta)")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)

    def _plot_mcc_row(ax_mcc, train_k: str, val_k: str, test_k: str, ylabel: str):
        ax_mcc.plot(
            ep,
            np.ma.masked_invalid(np.asarray(history[train_k], dtype=float)),
            "o-",
            label="Train",
            linewidth=1.2,
            markersize=3,
        )
        ax_mcc.plot(
            ep,
            np.ma.masked_invalid(np.asarray(history[val_k], dtype=float)),
            "s-",
            label="Validació",
            linewidth=1.2,
            markersize=3,
        )
        if has_test:
            ax_mcc.plot(
                ep,
                np.ma.masked_invalid(np.asarray(history[test_k], dtype=float)),
                "^-",
                label="Test",
                linewidth=1.2,
                markersize=3,
            )
        ax_mcc.set_xlabel("Epoch")
        ax_mcc.set_ylabel(ylabel)
        ax_mcc.legend()
        ax_mcc.grid(True, alpha=0.3)

    _plot_mcc_row(axes[1, 0], "train_mcc_micro", "val_mcc_micro", "test_mcc_micro", "MCC micro")
    _plot_mcc_row(axes[1, 1], "train_mcc_macro", "val_mcc_macro", "test_mcc_macro", "MCC macro")

    fig.tight_layout()
    path = os.path.join(figures_dir, "training_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"Gràfic d'entrenament desat: {path}")
    return path


# Import data loader utilities
try:
    from data_loader import create_data_loaders
except ImportError:
    # Fallback if data_loader is not available
    create_data_loaders = None


class TokenSequenceClassifier(nn.Module):
    """
    Model that processes a sequence of tokens and predicts a binary output.
    
    Architecture:
    1. Token embedding via lookup table
    2. Permutation-invariant self-attention layers
    3. Attention pooling to aggregate sequence
    4. MLP layers for binary classification
    """
    
    def __init__(
        self,
        K: int,  # Number of unique tokens (vocabulary size)
        dpeak: int,  # Embedding dimension
        L: int,  # Maximum sequence length
        num_attention_layers: int = 2,
        num_heads: int = 8,
        mlp_hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize the model.
        
        Args:
            K: Number of unique tokens (vocabulary size, tokens indexed 0 to K-1)
            dpeak: Embedding dimension for each token
            L: Maximum sequence length
            num_attention_layers: Number of self-attention layers
            num_heads: Number of attention heads
            mlp_hidden_dims: List of hidden dimensions for MLP layers
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super(TokenSequenceClassifier, self).__init__()
        
        self.K = K
        self.dpeak = dpeak
        self.L = L
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        
        # Token embedding lookup table
        self.token_embedding = nn.Embedding(K, dpeak)
        
        # Self-attention layers (permutation-invariant)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dpeak,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_attention_layers)
        ])
        
        # Layer normalization and feed-forward networks for each attention layer
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(dpeak)
                for _ in range(num_attention_layers)
            ])
        else:
            self.layer_norms = None
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dpeak, dpeak * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dpeak * 4, dpeak),
                nn.Dropout(dropout)
            )
            for _ in range(num_attention_layers)
        ])
        
        # Attention pooling: learnable query to aggregate sequence
        self.pooling_query = nn.Parameter(torch.randn(1, 1, dpeak))
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=dpeak,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP layers for binary classification
        mlp_layers = []
        prev_dim = dpeak
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Binary output layer
        mlp_layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, token_indices, attention_mask=None):
        """
        Forward pass.
        
        Args:
            token_indices: Input tensor of shape (batch_size, L) with token indices (0 to K-1)
                          Padded sequences should use a padding token (typically 0)
            attention_mask: Optional mask tensor of shape (batch_size, L) where 1 indicates
                          valid tokens and 0 indicates padding. If None, assumes padding is 0.
        
        Returns:
            Binary logits tensor of shape (batch_size, 1)
        """
        batch_size = token_indices.size(0)
        
        # Embed tokens: (batch_size, L) -> (batch_size, L, dpeak)
        x = self.token_embedding(token_indices)
        
        # Create attention mask if not provided (assume padding token is 0)
        if attention_mask is None:
            attention_mask = (token_indices != 0).float()  # 1 for valid tokens, 0 for padding
        
        # Convert to key_padding_mask: True = ignore (padding), False = attend
        # Shape: (batch_size, L)
        key_padding_mask = (attention_mask == 0).bool()  # True for padding positions
        
        # Apply self-attention layers (permutation-invariant)
        for i, attention_layer in enumerate(self.attention_layers):
            # Self-attention
            attn_output, _ = attention_layer(
                x, x, x,
                key_padding_mask=key_padding_mask if i == 0 else None  # Only apply mask in first layer
            )
            
            # Residual connection and layer norm
            if self.layer_norms is not None:
                x = self.layer_norms[i](x + attn_output)
            else:
                x = x + attn_output
            
            # Feed-forward network
            ff_output = self.ff_layers[i](x)
            
            # Residual connection
            if self.layer_norms is not None:
                x = self.layer_norms[i](x + ff_output)
            else:
                x = x + ff_output
        
        # Attention pooling: aggregate sequence into single vector
        # Expand pooling query to batch size
        query = self.pooling_query.expand(batch_size, -1, -1)  # (batch_size, 1, dpeak)
        
        # Apply attention pooling
        # Use key_padding_mask to ignore padding tokens
        pooled_output, _ = self.pooling_attention(
            query, x, x,
            key_padding_mask=key_padding_mask
        )
        
        # Squeeze sequence dimension: (batch_size, 1, dpeak) -> (batch_size, dpeak)
        pooled_output = pooled_output.squeeze(1)
        
        # MLP layers for binary classification
        logits = self.mlp(pooled_output)
        
        return logits
    
    def _train_epoch(self, train_loader, device, criterion, optimizer):
        """Train the model for one epoch."""
        self.training = True
        total_loss = 0.0
        correct = 0
        total = 0
        y_true_batches = []
        y_pred_batches = []
        
        for batch in tqdm(train_loader, desc="Training"):
            token_indices = batch['token_indices'].to(device)
            labels = batch['label'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self(token_indices, attention_mask).squeeze(-1)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            y_true_batches.append(labels.detach().cpu().numpy())
            y_pred_batches.append(predictions.detach().cpu().numpy())
            
            total_loss += loss.item()
        
        n_batches = len(train_loader)
        average_loss = total_loss / n_batches if n_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        if y_true_batches:
            y_true_all = np.concatenate(y_true_batches, axis=0)
            y_pred_all = np.concatenate(y_pred_batches, axis=0)
            precision, recall = _binary_precision_recall(y_true_all, y_pred_all)
            mcc = _safe_binary_mcc(y_true_all, y_pred_all)
        else:
            precision, recall, mcc = 0.0, 0.0, float("nan")
        
        return average_loss, accuracy, precision, recall, mcc
    
    def _evaluate(self, data_loader, device, criterion):
        """Evaluate the model on a dataset."""
        self.training = False
        total_loss = 0.0
        correct = 0
        total = 0
        y_true_batches = []
        y_pred_batches = []
        
        with torch.no_grad():
            for batch in data_loader:
                token_indices = batch['token_indices'].to(device)
                labels = batch['label'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                logits = self(token_indices, attention_mask).squeeze(-1)
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                y_true_batches.append(labels.detach().cpu().numpy())
                y_pred_batches.append(predictions.detach().cpu().numpy())
        
        n_batches = len(data_loader)
        average_loss = total_loss / n_batches if n_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        if y_true_batches:
            y_true_all = np.concatenate(y_true_batches, axis=0)
            y_pred_all = np.concatenate(y_pred_batches, axis=0)
            precision, recall = _binary_precision_recall(y_true_all, y_pred_all)
            mcc = _safe_binary_mcc(y_true_all, y_pred_all)
        else:
            precision, recall, mcc = 0.0, 0.0, float("nan")
        
        return average_loss, accuracy, precision, recall, mcc
    
    def train(
        self,
        train_data: Union[str, DataLoader],
        val_data: Union[str, DataLoader],
        test_data: Optional[Union[str, DataLoader]] = None,
        batch_size: int = 128,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
        output_dir: str = './checkpoints',
        log_dir: str = './logs',
        save_best: bool = False,
        verbose: bool = True,
        figures_train_dir: str = "figures_train",
    ):
        """
        Train the model.
        
        Args:
            train_data: Training data - either a file path (str) or DataLoader object
            val_data: Validation data - either a file path (str) or DataLoader object
            test_data: Optional test data - either a file path (str) or DataLoader object
            batch_size: Batch size (only used if file paths are provided)
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_workers: Number of data loader workers (only used if file paths are provided)
            device: Device to run on (default: cuda if available, else cpu)
            output_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            save_best: Save best model based on validation loss
            verbose: Print training progress
            figures_train_dir: Carpeta on desar training_summary.png al finalitzar (loss i accuracy).
        
        Returns:
            Training history dictionary with losses and accuracies
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if verbose:
            print(f'Using device: {device}')
        
        # Move model to device
        self.to(device)
        
        # Handle data loaders - convert file paths to DataLoaders if needed
        if isinstance(train_data, str):
            if create_data_loaders is None:
                raise ImportError("data_loader module is required when using file paths")
            if test_data is not None:
                train_loader, val_loader, test_loader = create_data_loaders(
                    train_data, val_data, test_data,
                    max_length=self.L,
                    batch_size=batch_size,
                    num_workers=num_workers
                )
            else:
                train_loader, val_loader = create_data_loaders(
                    train_data, val_data,
                    max_length=self.L,
                    batch_size=batch_size,
                    num_workers=num_workers
                )
                test_loader = None
        else:
            # Assume DataLoader objects
            train_loader = train_data
            val_loader = val_data
            test_loader = test_data
        
        if verbose:
            print(f'Training samples: {len(train_loader.dataset)}')
            print(f'Validation samples: {len(val_loader.dataset)}')
            if test_loader:
                print(f'Test samples: {len(test_loader.dataset)}')
            
            # Count parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'Total parameters: {total_params:,}')
            print(f'Trainable parameters: {trainable_params:,}')
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # TensorBoard writer
        writer = SummaryWriter(log_dir) if verbose else None
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_mcc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_mcc': [],
            'test_loss': [],
            'test_acc': [],
            'test_precision': [],
            'test_recall': [],
            'test_mcc': [],
        }
        
        # Training loop
        best_val_loss = float('inf')
        
        if verbose:
            print('\nStarting training...')
        
        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'\nEpoch {epoch}/{epochs}')
                print('-' * 50)
            
            # Train
            train_loss, train_acc, train_precision, train_recall, train_mcc = self._train_epoch(
                train_loader, device, criterion, optimizer
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_mcc'].append(train_mcc)
            
            # Evaluate on validation set
            val_loss, val_acc, val_precision, val_recall, val_mcc = self._evaluate(
                val_loader, device, criterion
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_mcc'].append(val_mcc)
            
            # Evaluate on test set if provided
            if test_loader is not None:
                test_loss, test_acc, test_precision, test_recall, test_mcc = self._evaluate(
                    test_loader, device, criterion
                )
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                history['test_precision'].append(test_precision)
                history['test_recall'].append(test_recall)
                history['test_mcc'].append(test_mcc)
            else:
                test_loss, test_acc = None, None
                test_precision, test_recall, test_mcc = None, None, None
            
            # Log metrics
            if writer is not None:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
                writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                writer.add_scalar('Precision/Train', train_precision, epoch)
                writer.add_scalar('Precision/Validation', val_precision, epoch)
                writer.add_scalar('Recall/Train', train_recall, epoch)
                writer.add_scalar('Recall/Validation', val_recall, epoch)
                if not np.isnan(train_mcc):
                    writer.add_scalar('MCC/Train', train_mcc, epoch)
                if not np.isnan(val_mcc):
                    writer.add_scalar('MCC/Validation', val_mcc, epoch)
                if test_loader is not None:
                    writer.add_scalar('Loss/Test', test_loss, epoch)
                    writer.add_scalar('Accuracy/Test', test_acc, epoch)
                    writer.add_scalar('Precision/Test', test_precision, epoch)
                    writer.add_scalar('Recall/Test', test_recall, epoch)
                    if not np.isnan(test_mcc):
                        writer.add_scalar('MCC/Test', test_mcc, epoch)
            
            # Print metrics
            if verbose:
                print(
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, MCC: {_format_mcc(train_mcc)}'
                )
                print(
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                    f'Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, MCC: {_format_mcc(val_mcc)}'
                )
                if test_loader is not None:
                    print(
                        f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, '
                        f'Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, MCC: {_format_mcc(test_mcc)}'
                    )
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_mcc': train_mcc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_mcc': val_mcc,
            }
            if test_loader is not None:
                checkpoint['test_loss'] = test_loss
                checkpoint['test_acc'] = test_acc
                checkpoint['test_precision'] = test_precision
                checkpoint['test_recall'] = test_recall
                checkpoint['test_mcc'] = test_mcc
            
            # Save latest checkpoint
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoint_multi_latest.pth'))
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(output_dir, 'checkpoint_multi_best.pth'))
                if verbose:
                    print(f'Saved best model (val_loss: {val_loss:.4f})')

        # Calibratge binari: umbral sobre σ(logits) perquè freq(pred=1) ~ freq(real=1) al train
        probs_train, y_train = _gather_probs_and_labels_binary(self, train_loader, device)
        train_pos_rate = float(np.mean(y_train)) if y_train.size > 0 else 0.0
        calibrated_threshold = _binary_threshold_for_positive_rate(probs_train, train_pos_rate)
        pred05 = (probs_train > 0.5).astype(np.int64) if probs_train.size > 0 else np.array([], dtype=np.int64)
        pred_cal = (probs_train > calibrated_threshold).astype(np.int64) if probs_train.size > 0 else np.array([], dtype=np.int64)
        err05 = float(np.mean(np.abs(pred05 - y_train))) if y_train.size > 0 else 0.0
        err_cal = float(np.mean(np.abs(pred_cal - y_train))) if y_train.size > 0 else 0.0

        def _inject_single_calibration_into_ckpt(path: str) -> None:
            if not os.path.isfile(path):
                return
            try:
                ckpt = torch.load(path, map_location="cpu")
                ckpt["sigmoid_threshold"] = float(calibrated_threshold)
                ckpt["calibration_target_pos_rate"] = float(train_pos_rate)
                ckpt["calibration_mode"] = "match_train_positive_rate"
                torch.save(ckpt, path)
            except OSError:
                pass

        _inject_single_calibration_into_ckpt(os.path.join(output_dir, "checkpoint_multi_latest.pth"))
        _inject_single_calibration_into_ckpt(os.path.join(output_dir, "checkpoint_multi_best.pth"))
        if verbose:
            print(
                "Calibratge binari desat al checkpoint | "
                f"tau={calibrated_threshold:.6f}, p_train(y=1)={train_pos_rate:.4f}, "
                f"error train {err_cal:.4f} (calibrat) vs {err05:.4f} (0.5)"
            )
        
        if writer is not None:
            writer.close()
        
        _save_training_figure_binary(
            history,
            figures_train_dir,
            title="Entrenament (classificació binària)",
            verbose=verbose,
        )

        if verbose:
            print('\nTraining completed!')
        
        return history
    
    def predict(
        self,
        input_data: Union[str, list],
        output_file: Optional[str] = None,
        batch_size: int = 128,
        device: Optional[torch.device] = None,
        pad_token: int = 0,
        verbose: bool = True,
        threshold: float = 0.5,
    ):
        """
        Predict labels for token sequences.
        
        Args:
            input_data: Either a file path (str) with token sequences, or a list of token sequences
                       File format: token1,token2,token3,... (one per line) or label;token1,token2,...
            output_file: Optional path to save predictions. If None, only returns predictions.
                        Format: label;token1,token2,token3,...
            batch_size: Batch size for prediction
            device: Device to run on (default: cuda if available, else cpu)
            pad_token: Token ID to use for padding (default: 0)
            verbose: Print prediction progress
            threshold: Umbral sobre σ(logits) per decidir classe 1.
        
        Returns:
            predictions: List of predicted labels (0 or 1)
            probabilities: List of prediction probabilities (0 to 1)
            sequences: List of input token sequences (if input_data is file path)
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device and set to eval mode
        self.to(device)
        self.training = False

        # Read sequences from file or use provided list
        if isinstance(input_data, str):
            sequences = []
            with open(input_data, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse tokens
                    if ',' in line:
                        tokens = [int(t.strip()) for t in line.split(',') if t.strip()]
                    else:
                        # Handle format with label (if present, ignore it)
                        if ';' in line:
                            parts = line.split(';', 1)
                            token_str = parts[1] if len(parts) > 1 else parts[0]
                        else:
                            token_str = line
                        
                        if token_str:
                            tokens = [int(t.strip()) for t in token_str.split(',') if t.strip()]
                        else:
                            tokens = []
                    
                    sequences.append(tokens)
            
            if verbose:
                print(f'Loaded {len(sequences)} sequences from {input_data}')
        else:
            # Assume input_data is a list of token sequences
            sequences = input_data
            if verbose:
                print(f'Processing {len(sequences)} sequences')
        
        # Process sequences in batches
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i+batch_size]
                
                # Pad sequences to max_length
                batch_tokens = []
                batch_masks = []
                
                for tokens in batch_sequences:
                    # Truncate if too long
                    if len(tokens) > self.L:
                        tokens = tokens[:self.L]
                    
                    # Pad to max_length
                    padded_tokens = tokens + [pad_token] * (self.L - len(tokens))
                    attention_mask = [1.0] * len(tokens) + [0.0] * (self.L - len(tokens))
                    
                    batch_tokens.append(padded_tokens)
                    batch_masks.append(attention_mask)
                
                # Convert to tensors
                token_indices = torch.tensor(batch_tokens, dtype=torch.long).to(device)
                attention_mask = torch.tensor(batch_masks, dtype=torch.float).to(device)
                
                # Predict
                logits = self(token_indices, attention_mask).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                batch_predictions = (probs > threshold).astype(int)
                
                predictions.extend(batch_predictions.tolist())
                probabilities.extend(probs.tolist())
        
        # Write predictions to output file if provided
        if output_file is not None:
            with open(output_file, 'w') as f:
                for tokens, pred in zip(sequences, predictions):
                    token_str = ','.join(map(str, tokens))
                    f.write(f'{pred};{token_str}\n')
            
            if verbose:
                print(f'Predictions written to {output_file}')
        
        # Print summary statistics
        if verbose:
            print(f'Total sequences: {len(predictions)}')
            print(f'Predicted class 0: {predictions.count(0)} ({100*predictions.count(0)/len(predictions):.1f}%)')
            print(f'Predicted class 1: {predictions.count(1)} ({100*predictions.count(1)/len(predictions):.1f}%)')
            print(f'Average probability: {np.mean(probabilities):.4f}')
            print(f'Min probability: {np.min(probabilities):.4f}')
            print(f'Max probability: {np.max(probabilities):.4f}')
        
        return predictions, probabilities, sequences


class TokenSequenceClassifierMulti(nn.Module):
    """
    Model that processes a sequence of tokens and predicts M binary outputs.
    
    Architecture:
    1. Token embedding via lookup table
    2. Permutation-invariant self-attention layers (shared across all labels)
    3. M separate attention pooling layers (one per label)
    4. M separate MLP layers for binary classification (one per label)
    """
    
    def __init__(
        self,
        K: int,  # Number of unique tokens (vocabulary size)
        dpeak: int,  # Embedding dimension
        L: int,  # Maximum sequence length
        M: int,  # Number of binary outputs (labels)
        num_attention_layers: int = 2,
        num_heads: int = 8,
        mlp_hidden_dims: list = [32, 16],
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize the multi-label model.
        
        Args:
            K: Number of unique tokens (vocabulary size, tokens indexed 0 to K-1)
            dpeak: Embedding dimension for each token
            L: Maximum sequence length
            M: Number of binary outputs (fixed, not padded)
            num_attention_layers: Number of self-attention layers
            num_heads: Number of attention heads
            mlp_hidden_dims: List of hidden dimensions for MLP layers
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super(TokenSequenceClassifierMulti, self).__init__()
        
        self.K = K
        self.dpeak = dpeak
        self.L = L
        self.M = M
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.fingerprint_bit_indices: Optional[List[int]] = None  # omple train() si subconjunt de bits
        
        # Token embedding lookup table (shared)
        self.token_embedding = nn.Embedding(K, dpeak)
        
        # Self-attention layers (permutation-invariant, shared)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dpeak,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_attention_layers)
        ])
        
        # Layer normalization and feed-forward networks for each attention layer
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(dpeak)
                for _ in range(num_attention_layers)
            ])
        else:
            self.layer_norms = None
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dpeak, dpeak * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dpeak * 4, dpeak),
                nn.Dropout(dropout)
            )
            for _ in range(num_attention_layers)
        ])
        
        # M separate attention pooling queries (one per label)
        self.pooling_queries = nn.Parameter(torch.randn(M, 1, dpeak))
        
        # M separate attention pooling layers (one per label)
        self.pooling_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dpeak,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(M)
        ])
        
        # M separate MLP layers for binary classification (one per label)
        self.mlps = nn.ModuleList()
        for _ in range(M):
            mlp_layers = []
            prev_dim = dpeak
            
            for hidden_dim in mlp_hidden_dims:
                mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            # Binary output layer
            mlp_layers.append(nn.Linear(prev_dim, 1))
            
            self.mlps.append(nn.Sequential(*mlp_layers))
    
    def forward(self, token_indices, attention_mask=None):
        """
        Forward pass.
        
        Args:
            token_indices: Input tensor of shape (batch_size, L) with token indices (0 to K-1)
                          Padded sequences should use a padding token (typically 0)
            attention_mask: Optional mask tensor of shape (batch_size, L) where 1 indicates
                          valid tokens and 0 indicates padding. If None, assumes padding is 0.
        
        Returns:
            Binary logits tensor of shape (batch_size, M)
        """
        batch_size = token_indices.size(0)
        
        # Embed tokens: (batch_size, L) -> (batch_size, L, dpeak)
        x = self.token_embedding(token_indices)
        
        # Create attention mask if not provided (assume padding token is 0)
        if attention_mask is None:
            attention_mask = (token_indices != 0).float()  # 1 for valid tokens, 0 for padding
        
        # Convert to key_padding_mask: True = ignore (padding), False = attend
        # Shape: (batch_size, L)
        key_padding_mask = (attention_mask == 0).bool()  # True for padding positions
        
        # Apply self-attention layers (permutation-invariant, shared)
        for i, attention_layer in enumerate(self.attention_layers):
            # Self-attention
            attn_output, _ = attention_layer(
                x, x, x,
                key_padding_mask=key_padding_mask if i == 0 else None  # Only apply mask in first layer
            )
            
            # Residual connection and layer norm
            if self.layer_norms is not None:
                x = self.layer_norms[i](x + attn_output)
            else:
                x = x + attn_output
            
            # Feed-forward network
            ff_output = self.ff_layers[i](x)
            
            # Residual connection
            if self.layer_norms is not None:
                x = self.layer_norms[i](x + ff_output)
            else:
                x = x + ff_output
        
        # M separate attention pooling layers (one per label)
        # Each pooling independently aggregates the sequence
        all_logits = []
        for m in range(self.M):
            # Get pooling query for this label: (1, 1, dpeak) -> (batch_size, 1, dpeak)
            query = self.pooling_queries[m:m+1].expand(batch_size, -1, -1)
            
            # Apply attention pooling for this label
            pooled_output, _ = self.pooling_attentions[m](
                query, x, x,
                key_padding_mask=key_padding_mask
            )
            
            # Squeeze sequence dimension: (batch_size, 1, dpeak) -> (batch_size, dpeak)
            pooled_output = pooled_output.squeeze(1)
            
            # MLP for this label: (batch_size, dpeak) -> (batch_size, 1)
            logits = self.mlps[m](pooled_output)
            all_logits.append(logits)
        
        # Concatenate all logits: [(batch_size, 1), ...] -> (batch_size, M)
        logits = torch.cat(all_logits, dim=1)
        
        return logits
    
    def _train_epoch(
        self,
        train_loader,
        device,
        criterion,
        optimizer,
        target_marginal_freqs: Optional[torch.Tensor] = None,
        calibration_lambda: float = 0.0,
    ):
        """Train the model for one epoch."""
        self.training = True
        total_loss = 0.0
        correct = 0
        total = 0
        y_true_batches = []
        y_pred_batches = []

        for batch in tqdm(train_loader, desc="Training"):
            token_indices = batch['token_indices'].to(device)
            labels = batch['labels'].to(device)  # Shape: (batch_size, M)
            attention_mask = batch['attention_mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self(token_indices, attention_mask)  # Shape: (batch_size, M)
            
            # Calculate loss
            loss = criterion(logits, labels)
            if calibration_lambda > 0.0 and target_marginal_freqs is not None:
                batch_marginal = torch.sigmoid(logits).mean(dim=0)
                loss = loss + calibration_lambda * F.mse_loss(
                    batch_marginal, target_marginal_freqs
                )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Calculate accuracy (per-label accuracy: average across all labels)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == labels).sum().item()  # Count all correct label predictions
            total += labels.size(0) * labels.size(1)  # Total number of labels (batch_size * M)
            
            total_loss += loss.item()
            y_true_batches.append(labels.detach().cpu().numpy())
            y_pred_batches.append(predictions.detach().cpu().numpy())
        
        average_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        if y_true_batches:
            y_true_all = np.concatenate(y_true_batches, axis=0)
            y_pred_all = np.concatenate(y_pred_batches, axis=0)
            mcc_micro, mcc_macro = _multilabel_mcc_micro_macro(y_true_all, y_pred_all)
        else:
            mcc_micro, mcc_macro = float("nan"), float("nan")

        return average_loss, accuracy, mcc_micro, mcc_macro
    
    def _evaluate(
        self,
        data_loader,
        device,
        criterion,
        per_bit_threshold: Optional[torch.Tensor] = None,
    ):
        """Evaluate the model on a dataset."""
        self.training = False
        total_loss = 0.0
        correct = 0
        total = 0
        y_true_batches = []
        y_pred_batches = []

        with torch.no_grad():
            for batch in data_loader:
                token_indices = batch['token_indices'].to(device)
                labels = batch['labels'].to(device)  # Shape: (batch_size, M)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                logits = self(token_indices, attention_mask)  # Shape: (batch_size, M)
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                if per_bit_threshold is not None:
                    predictions = (probs > per_bit_threshold.unsqueeze(0)).float()
                else:
                    predictions = (probs > 0.5).float()
                correct += (predictions == labels).sum().item()  # Count all correct label predictions
                total += labels.size(0) * labels.size(1)  # Total number of labels (batch_size * M)

                y_true_batches.append(labels.detach().cpu().numpy())
                y_pred_batches.append(predictions.detach().cpu().numpy())
        
        n_batches = len(data_loader)
        average_loss = total_loss / n_batches if n_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        if y_true_batches:
            y_true_all = np.concatenate(y_true_batches, axis=0)
            y_pred_all = np.concatenate(y_pred_batches, axis=0)
            mcc_micro, mcc_macro = _multilabel_mcc_micro_macro(y_true_all, y_pred_all)
        else:
            mcc_micro, mcc_macro = float("nan"), float("nan")

        return average_loss, accuracy, mcc_micro, mcc_macro
    
    def train(
        self,
        train_data: Union[str, DataLoader],
        val_data: Union[str, DataLoader],
        test_data: Optional[Union[str, DataLoader]] = None,
        batch_size: int = 128,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_workers: int = 0,
        device: Optional[torch.device] = None,
        output_dir: str = './checkpoints',
        log_dir: str = './logs',
        save_best: bool = False,
        verbose: bool = True,
        label_indices: Optional[List[int]] = None,
        full_label_dim: int = 2048,
        figures_train_dir: str = "figures_train",
        calibration_lambda: float = 0.25,
    ):
        """
        Train the multi-label model.
        
        Args:
            train_data: Training data - either a file path (str) or DataLoader object
            val_data: Validation data - either a file path (str) or DataLoader object
            test_data: Optional test data - either a file path (str) or DataLoader object
            batch_size: Batch size (only used if file paths are provided)
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_workers: Number of data loader workers (only used if file paths are provided)
            device: Device to run on (default: cuda if available, else cpu)
            output_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            save_best: Save best model based on validation loss
            verbose: Print training progress
            label_indices: Índexs 0-based dels bits al fingerprint complet del fitxer (p. ex. Morgan 2048).
                Si no és None, el fitxer ha de tenir `full_label_dim` etiquetes per línia i només
                s'entrenen aquestes columnes; self.M ha de ser len(label_indices).
            full_label_dim: Nombre d'etiquetes per línia al fitxer quan s'usa label_indices.
            figures_train_dir: Carpeta on desar training_summary.png (loss, accuracy, MCC).
            calibration_lambda: Pes de la pèrdua auxiliar que alinea la mitjana de σ(logits)
                per bit amb la freqüència empírica de y=1 al train (calibratge marginal).
                0 desactiva aquest terme; es desa igualment la taula d'umbral τ per bit.
        
        Returns:
            Training history dictionary with losses and accuracies
        """
        # Import here to avoid circular imports
        try:
            from data_loader import create_data_loaders_multi
        except ImportError:
            create_data_loaders_multi = None
        
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        _multi_train_title = f"Entrenament multi-etiqueta (M = {self.M})"
        
        if verbose:
            print(f'Using device: {device}')
            if matthews_corrcoef is None:
                print(
                    "Avís: scikit-learn no instal·lat — el MCC sortirà com a n/a. "
                    "Instal·la amb: py -m pip install scikit-learn"
                )
        
        # Move model to device
        self.to(device)

        if label_indices is not None:
            if len(label_indices) != self.M:
                raise ValueError(
                    f"len(label_indices)={len(label_indices)} ha de coincidir amb el M del model ({self.M})"
                )
            self.fingerprint_bit_indices = list(label_indices)
        else:
            self.fingerprint_bit_indices = None
        
        # Handle data loaders - convert file paths to DataLoaders if needed
        if isinstance(train_data, str):
            if create_data_loaders_multi is None:
                raise ImportError("data_loader module is required when using file paths")
            if test_data is not None:
                train_loader, val_loader, test_loader = create_data_loaders_multi(
                    train_data, val_data, test_data,
                    max_length=self.L,
                    M=self.M,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    label_indices=label_indices,
                    full_label_dim=full_label_dim,
                )
            else:
                train_loader, val_loader = create_data_loaders_multi(
                    train_data, val_data,
                    max_length=self.L,
                    M=self.M,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    label_indices=label_indices,
                    full_label_dim=full_label_dim,
                )
                test_loader = None
        else:
            # Assume DataLoader objects
            train_loader = train_data
            val_loader = val_data
            test_loader = test_data
        
        if verbose:
            print(f'Training samples: {len(train_loader.dataset)}')
            print(f'Validation samples: {len(val_loader.dataset)}')
            if test_loader:
                print(f'Test samples: {len(test_loader.dataset)}')
            
            # Count parameters
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'Total parameters: {total_params:,}')
            print(f'Trainable parameters: {trainable_params:,}')
            print(f'Number of labels (M): {self.M}')
            if self.fingerprint_bit_indices is not None:
                print(
                    f"Subconjunt de bits al fingerprint ({full_label_dim} posicions al fitxer): "
                    f"{len(self.fingerprint_bit_indices)} índexs"
                )
                preview = self.fingerprint_bit_indices[:20]
                more = " ..." if len(self.fingerprint_bit_indices) > 20 else ""
                print(f"  Índexs (mostra): {preview}{more}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        all_labels = torch.tensor(train_loader.dataset.labels, dtype=torch.float32)
        freqs = all_labels.mean(dim=0) # Freqüència de cada bit (0 a 1)
        target_marginal_freqs = freqs.to(device)

        # Pes invers per bit: com menys freqüent és un bit, més car resulta equivocar-s'hi.
        # pos_weights és un vector de M posicions (una per bit entrenat).
        pos_weights = (1.0 / (freqs + 1e-6)) * 10

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
        optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # TensorBoard writer
        writer = SummaryWriter(log_dir) if verbose else None
        
        # Training history (MCC micro/macro per a classificació multi-etiqueta)
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_mcc_micro': [],
            'train_mcc_macro': [],
            'val_loss': [],
            'val_acc': [],
            'val_mcc_micro': [],
            'val_mcc_macro': [],
            'test_loss': [],
            'test_acc': [],
            'test_mcc_micro': [],
            'test_mcc_macro': [],
        }
        
        # Training loop
        best_val_loss = float('inf')
        
        if verbose:
            print('\nStarting training...')
            if calibration_lambda > 0:
                print(
                    f"Calibratge marginal actiu: λ={calibration_lambda} "
                    "(MSE entre mitjana batch σ(logits) i freq. y=1 al train)"
                )

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f'\nEpoch {epoch}/{epochs}')
                print('-' * 50)
            
            # Train
            train_loss, train_acc, train_mcc_micro, train_mcc_macro = self._train_epoch(
                train_loader,
                device,
                criterion,
                optimizer,
                target_marginal_freqs=target_marginal_freqs,
                calibration_lambda=calibration_lambda,
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_mcc_micro'].append(train_mcc_micro)
            history['train_mcc_macro'].append(train_mcc_macro)
            
            # Evaluate on validation set
            val_loss, val_acc, val_mcc_micro, val_mcc_macro = self._evaluate(
                val_loader, device, criterion
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_mcc_micro'].append(val_mcc_micro)
            history['val_mcc_macro'].append(val_mcc_macro)
            
            # Evaluate on test set if provided
            if test_loader is not None:
                test_loss, test_acc, test_mcc_micro, test_mcc_macro = self._evaluate(
                    test_loader, device, criterion
                )
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                history['test_mcc_micro'].append(test_mcc_micro)
                history['test_mcc_macro'].append(test_mcc_macro)
            else:
                test_loss, test_acc = None, None
                test_mcc_micro, test_mcc_macro = None, None
            
            # Log metrics (TensorBoard no accepta bé NaN als escalars)
            if writer is not None:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
                writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                if not np.isnan(train_mcc_micro):
                    writer.add_scalar('MCC_micro/Train', train_mcc_micro, epoch)
                if not np.isnan(train_mcc_macro):
                    writer.add_scalar('MCC_macro/Train', train_mcc_macro, epoch)
                if not np.isnan(val_mcc_micro):
                    writer.add_scalar('MCC_micro/Validation', val_mcc_micro, epoch)
                if not np.isnan(val_mcc_macro):
                    writer.add_scalar('MCC_macro/Validation', val_mcc_macro, epoch)
                if test_loader is not None:
                    writer.add_scalar('Loss/Test', test_loss, epoch)
                    writer.add_scalar('Accuracy/Test', test_acc, epoch)
                    if not np.isnan(test_mcc_micro):
                        writer.add_scalar('MCC_micro/Test', test_mcc_micro, epoch)
                    if not np.isnan(test_mcc_macro):
                        writer.add_scalar('MCC_macro/Test', test_mcc_macro, epoch)
            
            # Print metrics
            if verbose:
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'MCC micro: {_format_mcc(train_mcc_micro)}, MCC macro: {_format_mcc(train_mcc_macro)}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                      f'MCC micro: {_format_mcc(val_mcc_micro)}, MCC macro: {_format_mcc(val_mcc_macro)}')
                if test_loader is not None:
                    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, '
                          f'MCC micro: {_format_mcc(test_mcc_micro)}, MCC macro: {_format_mcc(test_mcc_macro)}')
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_mcc_micro': train_mcc_micro,
                'train_mcc_macro': train_mcc_macro,
                'val_mcc_micro': val_mcc_micro,
                'val_mcc_macro': val_mcc_macro,
                'M': self.M,
                'fingerprint_bit_indices': self.fingerprint_bit_indices,
                'full_label_dim': full_label_dim if self.fingerprint_bit_indices else None,
            }
            if test_loader is not None:
                checkpoint['test_loss'] = test_loss
                checkpoint['test_acc'] = test_acc
                checkpoint['test_mcc_micro'] = test_mcc_micro
                checkpoint['test_mcc_macro'] = test_mcc_macro
            
            # Save latest checkpoint
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoint_latest.pth'))
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(output_dir, 'checkpoint_best.pth'))
                if verbose:
                    print(f'Saved best model (val_loss: {val_loss:.4f})')
        
        # Calibratge per bit: umbral τ sobre σ per igualar freq. de prediccions 1 a la marginal
        probs_train, _ = _gather_probs_and_labels_multilabel(self, train_loader, device)
        freqs_np = freqs.cpu().numpy()
        thresholds_np = _per_bit_thresholds_for_positive_rate(probs_train, freqs_np)
        mean_prob = probs_train.mean(axis=0)
        freq_pred_05 = (probs_train > 0.5).mean(axis=0)
        freq_pred_cal = (probs_train > thresholds_np.reshape(1, -1)).mean(axis=0)
        fp_positions = (
            self.fingerprint_bit_indices if self.fingerprint_bit_indices is not None else None
        )
        _save_calibration_table_multilabel(
            freqs_np,
            freq_pred_05,
            freq_pred_cal,
            mean_prob,
            thresholds_np,
            fp_positions,
            figures_train_dir,
            verbose=verbose,
        )
        tau_tensor = torch.tensor(thresholds_np, dtype=torch.float32)
        ckpt_path_latest = os.path.join(output_dir, "checkpoint_latest.pth")
        def _inject_calibration_into_ckpt(path: str) -> None:
            if not os.path.isfile(path):
                return
            try:
                ckpt = torch.load(path, map_location="cpu")
                ckpt["per_bit_sigmoid_threshold"] = tau_tensor
                ckpt["calibration_target_freq"] = freqs
                ckpt["calibration_lambda_used"] = calibration_lambda
                torch.save(ckpt, path)
            except OSError:
                pass

        _inject_calibration_into_ckpt(ckpt_path_latest)
        _inject_calibration_into_ckpt(os.path.join(output_dir, "checkpoint_best.pth"))
        if verbose and os.path.isfile(ckpt_path_latest):
            err_cal = float(np.mean(np.abs(freq_pred_cal - freqs_np)))
            err05 = float(np.mean(np.abs(freq_pred_05 - freqs_np)))
            print(
                f"Checkpoints amb τ per bit | error mitjà |freq_pred−freq_true|: "
                f"{err_cal:.4f} (calibrat) vs {err05:.4f} (umbral 0.5)"
            )

        if writer is not None:
            writer.close()
        
        _save_training_figure_multilabel(
            history,
            figures_train_dir,
            title=_multi_train_title,
            verbose=verbose,
        )

        if verbose:
            print('\nTraining completed!')
        
        return history
    
    def predict(
        self,
        input_data: Union[str, list],
        output_file: Optional[str] = None,
        batch_size: int = 128,
        device: Optional[torch.device] = None,
        pad_token: int = 0,
        verbose: bool = True,
        per_bit_threshold: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Predict labels for token sequences (multi-label).
        
        Args:
            input_data: Either a file path (str) with token sequences, or a list of token sequences
                       File format: token1,token2,token3,... (one per line) or label1,label2,...,labelM;token1,token2,...
            output_file: Optional path to save predictions. If None, only returns predictions.
                        Format: label1,label2,...,labelM;token1,token2,token3,...
            batch_size: Batch size for prediction
            device: Device to run on (default: cuda if available, else cpu)
            pad_token: Token ID to use for padding (default: 0)
            verbose: Print prediction progress
            per_bit_threshold: Vector (M,) amb umbral sobre σ per cada bit (p.ex. checkpoint
                `per_bit_sigmoid_threshold`). None → umbral 0.5 per tots els bits.
        
        Returns:
            predictions: List of predicted label lists, each of length M (e.g., [[0,1,0], [1,0,1], ...])
            probabilities: List of probability lists, each of length M (e.g., [[0.1,0.9,0.2], ...])
            sequences: List of input token sequences (if input_data is file path)
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device and set to eval mode
        self.to(device)
        self.training = False

        # Read sequences from file or use provided list
        if isinstance(input_data, str):
            sequences = []
            with open(input_data, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse tokens - handle format with or without labels
                    if ';' in line:
                        parts = line.split(';', 1)
                        token_str = parts[1] if len(parts) > 1 else parts[0]
                    else:
                        token_str = line
                    
                    if token_str:
                        tokens = [int(t.strip()) for t in token_str.split(',') if t.strip()]
                    else:
                        tokens = []
                    
                    sequences.append(tokens)
            
            if verbose:
                print(f'Loaded {len(sequences)} sequences from {input_data}')
        else:
            # Assume input_data is a list of token sequences
            sequences = input_data
            if verbose:
                print(f'Processing {len(sequences)} sequences')
        
        # Process sequences in batches
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i+batch_size]
                
                # Pad sequences to max_length
                batch_tokens = []
                batch_masks = []
                
                for tokens in batch_sequences:
                    # Truncate if too long
                    if len(tokens) > self.L:
                        tokens = tokens[:self.L]
                    
                    # Pad to max_length
                    padded_tokens = tokens + [pad_token] * (self.L - len(tokens))
                    attention_mask = [1.0] * len(tokens) + [0.0] * (self.L - len(tokens))
                    
                    batch_tokens.append(padded_tokens)
                    batch_masks.append(attention_mask)
                
                # Convert to tensors
                token_indices = torch.tensor(batch_tokens, dtype=torch.long).to(device)
                attention_mask = torch.tensor(batch_masks, dtype=torch.float).to(device)
                
                # Predict
                logits = self(token_indices, attention_mask)  # Shape: (batch_size, M)
                probs = torch.sigmoid(logits).cpu().numpy()  # Shape: (batch_size, M)
                if per_bit_threshold is not None:
                    thr = np.asarray(per_bit_threshold, dtype=np.float64).reshape(1, -1)
                    if thr.shape[1] != probs.shape[1]:
                        raise ValueError(
                            "per_bit_threshold ha de tenir M elements (mateix nombre que columnes de probs)"
                        )
                    batch_predictions = (probs > thr).astype(int)
                else:
                    batch_predictions = (probs > 0.5).astype(int)
                
                # Convert to list of lists
                predictions.extend(batch_predictions.tolist())
                probabilities.extend(probs.tolist())
        
        # Write predictions to output file if provided
        if output_file is not None:
            with open(output_file, 'w') as f:
                for tokens, pred in zip(sequences, predictions):
                    token_str = ','.join(map(str, tokens))
                    pred_str = ','.join(map(str, pred))
                    f.write(f'{pred_str};{token_str}\n')
            
            if verbose:
                print(f'Predictions written to {output_file}')
        
        # Print summary statistics
        if verbose:
            all_probs = np.array(probabilities).flatten()
            all_preds = np.array(predictions).flatten()
            
            print(f'Total sequences: {len(predictions)}')
            print(f'Total labels (M={self.M}): {len(all_preds)}')
            print(f'Predicted class 0: {np.sum(all_preds == 0)} ({100*np.sum(all_preds == 0)/len(all_preds):.1f}%)')
            print(f'Predicted class 1: {np.sum(all_preds == 1)} ({100*np.sum(all_preds == 1)/len(all_preds):.1f}%)')
            
            # Per-label statistics
            if self.M > 1:
                print(f'\nPer-label statistics:')
                probs_array = np.array(probabilities)  # Shape: (num_sequences, M)
                for m in range(self.M):
                    label_probs = probs_array[:, m]
                    label_preds = np.array([p[m] for p in predictions])
                    print(f'  Label {m+1}: Predicted 1: {np.sum(label_preds == 1)}/{len(label_preds)} '
                          f'({100*np.sum(label_preds == 1)/len(label_preds):.1f}%), '
                          f'Avg prob: {np.mean(label_probs):.4f}')
            
            print(f'\nOverall average probability: {np.mean(all_probs):.4f}')
            print(f'Min probability: {np.min(all_probs):.4f}')
            print(f'Max probability: {np.max(all_probs):.4f}')
        
        return predictions, probabilities, sequences

