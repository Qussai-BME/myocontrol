"""
cnn_baseline.py - CNN-1D Baseline for EMG classification
MyoControl Suite v0.4

Implements the CNN-1D baseline matching the paper's architecture:
- 3 parallel convolutional branches (kernel 3/5/7, 32 filters each)
- Concatenation → BatchNorm
- 2 Conv1D blocks (64 filters)
- Global average pooling
- Softmax over classes

Reference: Adlbi & Darwich (2026) §3.5
"""
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. CNN-1D baseline disabled.")


if HAS_TORCH:

    class MultiScaleCNN1D(nn.Module):
        """
        Multi-scale CNN-1D (paper §3.5).

        Architecture:
            Input: (batch, channels, samples)
            → 3 parallel branches (kernel 3/5/7, 32 filters each)
            → BatchNorm
            → 2 Conv1D blocks (64 filters)
            → Global Average Pooling
            → Softmax
        """

        def __init__(self, n_channels: int = 12, n_classes: int = 41,
                     window_size: int = 800,  # 400ms @ 2000Hz
                     dropout: float = 0.3):
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.window_size = window_size

            # Multi-scale branches
            self.branch3 = self._make_branch(3, 32, n_channels)
            self.branch5 = self._make_branch(5, 32, n_channels)
            self.branch7 = self._make_branch(7, 32, n_channels)

            # BatchNorm after concatenation
            self.bn1 = nn.BatchNorm1d(96)  # 32*3

            # Conv blocks (64 filters)
            self.conv_block1 = self._make_conv_block(96, 64, kernel=3, dropout=dropout)
            self.conv_block2 = self._make_conv_block(64, 64, kernel=3, dropout=dropout)

            # Global average pooling + classifier
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(64, n_classes)

        def _make_branch(self, kernel: int, filters: int, in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel, padding=kernel // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        def _make_conv_block(self, in_ch: int, out_ch: int, kernel: int,
                              dropout: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
            )

        def forward(self, x):
            # x: (batch, n_channels, window_size)
            b3 = self.branch3(x)
            b5 = self.branch5(x)
            b7 = self.branch7(x)

            # Concatenate along channel dim
            x = torch.cat([b3, b5, b7], dim=1)
            x = self.bn1(x)

            x = self.conv_block1(x)
            x = self.conv_block2(x)

            x = self.gap(x).squeeze(-1)  # (batch, 64)
            return self.classifier(x)  # logits

    class CNN1DClassifier:
        """
        Sklearn-compatible wrapper around the PyTorch CNN-1D model.
        """

        def __init__(self, n_channels: int = 12, n_classes: int = 41,
                     window_size: int = 800,
                     n_epochs: int = 30, batch_size: int = 64,
                     lr: float = 1e-3, device: str = 'auto',
                     random_state: int = 42):
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.window_size = window_size
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.lr = lr
            self.random_state = random_state

            if device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device

            self.model: Optional[MultiScaleCNN1D] = None
            self.label_to_idx: Optional[Dict] = None
            self.idx_to_label: Optional[Dict] = None
            self.classes_: Optional[List] = None
            self.is_fitted = False

        def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
            """
            Convert (n_samples, n_features) to (n_samples, n_channels, window_size).
            For now, assumes X is already shaped (n_samples, n_channels, window_size).
            """
            if X.ndim == 2:
                # Assume features — reshape not possible, treat as flat signal
                # Use 1 channel = full feature vector
                X = X.reshape(X.shape[0], 1, X.shape[1])
            return torch.FloatTensor(X).to(self.device)

        def fit(self, X: np.ndarray, y: np.ndarray):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

            self.classes_ = sorted(np.unique(y).tolist())
            self.label_to_idx = {c: i for i, c in enumerate(self.classes_)}
            self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
            y_encoded = np.array([self.label_to_idx[c] for c in y])

            n_samples = X.shape[0]
            n_channels = X.shape[1] if X.ndim == 3 else 1

            self.model = MultiScaleCNN1D(
                n_channels=n_channels,
                n_classes=len(self.classes_),
                window_size=X.shape[-1] if X.ndim == 3 else X.shape[1],
            ).to(self.device)

            # Loss + optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            # Convert to tensors
            X_tensor = self._to_tensor(X)
            y_tensor = torch.LongTensor(y_encoded).to(self.device)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model.train()
            for epoch in range(self.n_epochs):
                total_loss = 0.0
                correct = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_X.size(0)
                    correct += (outputs.argmax(dim=1) == batch_y).sum().item()
                epoch_loss = total_loss / n_samples
                epoch_acc = correct / n_samples
                if (epoch + 1) % 5 == 0:
                    logger.info(f"  CNN epoch {epoch+1}/{self.n_epochs}: "
                                f"loss={epoch_loss:.4f}, acc={epoch_acc*100:.2f}%")

            self.is_fitted = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if not self.is_fitted:
                raise RuntimeError("CNN not fitted.")
            self.model.eval()
            X_tensor = self._to_tensor(X)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                preds = outputs.argmax(dim=1).cpu().numpy()
            return np.array([self.idx_to_label[int(p)] for p in preds])

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            if not self.is_fitted:
                raise RuntimeError("CNN not fitted.")
            self.model.eval()
            X_tensor = self._to_tensor(X)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs

        def count_parameters(self) -> int:
            if self.model is None:
                return 0
            return sum(p.numel() for p in self.model.parameters())

else:
    # Stub if PyTorch not available
    class CNN1DClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for CNN1DClassifier. "
                "Install with: pip install torch")
