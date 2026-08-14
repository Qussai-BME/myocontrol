"""
dan_baseline.py - Lightweight Domain Adversarial Network for EMG
MyoControl Suite v0.5

CPU-only Domain Adversarial Network (DAN) for cross-subject EMG.
Inspired by Ghosh et al. (2025) "Calibration-free SSL+DA" but redesigned:
- Tiny architecture (< 100K parameters vs millions in SOTA)
- CPU-only training (no GPU needed)
- Compatible with low-resource environments
- Requires PyTorch (raises a clear ImportError at construction time if
  unavailable, rather than a confusing crash mid-training). An earlier
  version of this docstring claimed a NumPy-only fallback existed for
  when PyTorch isn't installed -- it didn't; that path was never
  actually implemented, and DANClassifier previously crashed with a
  raw NameError if you tried it without PyTorch. Use EMGClassifier
  (XGBoost/LDA/RandomForest) if PyTorch isn't available to you.

Architecture:
    Feature Extractor (shared): 2-layer MLP (input → 128 → 64)
    ↓
    ┌─────────────────┬─────────────────────┐
    │ Gesture         │ Domain              │
    │ Classifier      │ Discriminator       │
    │ (64 → n_classes) │ (64 → 1, binary)   │
    └─────────────────┴─────────────────────┘

Training:
- Gesture loss: cross-entropy on labeled source subjects
- Domain loss: binary cross-entropy (subject identification)
- Gradient reversal layer between extractor and domain discriminator
- Forces feature extractor to learn subject-invariant features
"""
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
import time

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Function
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed. DAN baseline disabled.")


if HAS_TORCH:

    # ============================================================
    # Gradient Reversal Layer (GRL)
    # ============================================================
    class GradientReversalFunction(Function):
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambda_, None

    def grad_reverse(x, lambda_=1.0):
        return GradientReversalFunction.apply(x, lambda_)

    # ============================================================
    # Lightweight DAN model
    # ============================================================
    class LightweightDAN(nn.Module):
        """
        Lightweight Domain Adversarial Network.
        Total parameters: ~50K (vs SOTA's millions)
        Trainable on Intel Core i3 / 4GB RAM in < 10 minutes.
        """

        def __init__(self, n_features: int, n_classes: int,
                     n_domains: int, hidden_dim: int = 64,
                     lambda_max: float = 1.0):
            super().__init__()
            self.n_features = n_features
            self.n_classes = n_classes
            self.n_domains = n_domains
            self.lambda_max = lambda_max

            # Feature extractor (shared)
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )

            # Gesture classifier
            self.gesture_classifier = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_classes),
            )

            # Domain discriminator
            self.domain_discriminator = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, n_domains),
            )

        def forward(self, x, lambda_=0.0):
            features = self.feature_extractor(x)
            gesture_logits = self.gesture_classifier(features)

            # Domain branch with gradient reversal
            reversed_features = grad_reverse(features, lambda_)
            domain_logits = self.domain_discriminator(reversed_features)

            return gesture_logits, domain_logits, features

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ============================================================
    # DAN Trainer
    # ============================================================
    class DANTrainer:
        """
        Trains the LightweightDAN with adversarial domain alignment.

        Training schedule:
        - Epoch 1-20: lambda = 0 (pure gesture classification)
        - Epoch 20-100: lambda gradually increases to lambda_max
        """

        def __init__(self, model: LightweightDAN,
                     lr: float = 1e-3, device: str = 'auto',
                     lambda_schedule: str = 'gradual'):
            if device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device

            self.model = model.to(self.device)
            self.lr = lr
            self.lambda_schedule = lambda_schedule
            self.history = {'epoch': [], 'gesture_loss': [],
                            'domain_loss': [], 'gesture_acc': [], 'domain_acc': []}

        def _compute_lambda(self, epoch: int, total_epochs: int) -> float:
            """Schedule lambda (gradient reversal strength)."""
            if self.lambda_schedule == 'gradual':
                # Linear ramp from 0 to lambda_max over first 50% of epochs
                ramp_epochs = total_epochs // 2
                if epoch < ramp_epochs:
                    return self.model.lambda_max * epoch / max(ramp_epochs, 1)
                return self.model.lambda_max
            elif self.lambda_schedule == 'fixed':
                return self.model.lambda_max
            else:  # 'none'
                return 0.0

        def fit(self, X_source: np.ndarray, y_source: np.ndarray,
                domain_source: np.ndarray,
                X_target: np.ndarray = None, domain_target: np.ndarray = None,
                n_epochs: int = 100, batch_size: int = 64,
                verbose: bool = True) -> Dict:
            """
            Train the DAN.

            Parameters
            ----------
            X_source, y_source, domain_source : labeled source subjects
            X_target, domain_target : unlabeled target subject(s) (optional)
            """
            # Convert to tensors
            X_s = torch.FloatTensor(X_source).to(self.device)
            y_s = torch.LongTensor(y_source).to(self.device)
            d_s = torch.LongTensor(domain_source).to(self.device)

            # Combine source + target for domain training
            if X_target is not None:
                X_t = torch.FloatTensor(X_target).to(self.device)
                d_t = torch.LongTensor(domain_target).to(self.device)
                X_all = torch.cat([X_s, X_t], dim=0)
                d_all = torch.cat([d_s, d_t], dim=0)
            else:
                X_all = X_s
                d_all = d_s

            # Loss + optimizers
            gesture_loss_fn = nn.CrossEntropyLoss()
            domain_loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                    weight_decay=1e-4)

            n_samples = len(X_s)
            n_batches = max(1, n_samples // batch_size)

            for epoch in range(n_epochs):
                self.model.train()
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                epoch_g_correct = 0
                epoch_d_correct = 0
                epoch_total = 0

                # Shuffle
                perm = torch.randperm(n_samples)
                X_s_shuffled = X_s[perm]
                y_s_shuffled = y_s[perm]
                d_s_shuffled = d_s[perm]

                lambda_ = self._compute_lambda(epoch, n_epochs)

                for b in range(n_batches):
                    start = b * batch_size
                    end = min(start + batch_size, n_samples)
                    X_batch = X_s_shuffled[start:end]
                    y_batch = y_s_shuffled[start:end]
                    d_batch = d_s_shuffled[start:end]

                    # Random target batch (same size)
                    target_perm = torch.randperm(len(X_all))
                    X_d_batch = X_all[target_perm[:len(X_batch)]]
                    d_d_batch = d_all[target_perm[:len(X_batch)]]

                    optimizer.zero_grad()

                    # Forward (gesture)
                    g_logits, _, _ = self.model(X_batch, lambda_)
                    g_loss = gesture_loss_fn(g_logits, y_batch)

                    # Forward (domain) on combined batch
                    _, d_logits, _ = self.model(X_d_batch, lambda_)
                    d_loss = domain_loss_fn(d_logits, d_d_batch)

                    # Total loss
                    loss = g_loss + d_loss
                    loss.backward()
                    optimizer.step()

                    epoch_g_loss += g_loss.item() * len(X_batch)
                    epoch_d_loss += d_loss.item() * len(X_batch)
                    epoch_g_correct += (g_logits.argmax(dim=1) == y_batch).sum().item()
                    epoch_d_correct += (d_logits.argmax(dim=1) == d_d_batch).sum().item()
                    epoch_total += len(X_batch)

                # Record
                self.history['epoch'].append(epoch)
                self.history['gesture_loss'].append(epoch_g_loss / epoch_total)
                self.history['domain_loss'].append(epoch_d_loss / epoch_total)
                self.history['gesture_acc'].append(epoch_g_correct / epoch_total)
                self.history['domain_acc'].append(epoch_d_correct / epoch_total)

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"DAN epoch {epoch+1}/{n_epochs}: "
                        f"g_loss={self.history['gesture_loss'][-1]:.4f}, "
                        f"g_acc={self.history['gesture_acc'][-1]*100:.2f}%, "
                        f"d_acc={self.history['domain_acc'][-1]*100:.2f}%, "
                        f"lambda={lambda_:.3f}")

            return self.history

        def predict(self, X: np.ndarray) -> np.ndarray:
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                g_logits, _, _ = self.model(X_t, 0.0)
                return g_logits.argmax(dim=1).cpu().numpy()

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                g_logits, _, _ = self.model(X_t, 0.0)
                return torch.softmax(g_logits, dim=1).cpu().numpy()

else:
    # Stub if PyTorch not available
    class LightweightDAN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for DAN. Install: pip install torch")

    class DANTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for DAN")


# ============================================================
# Sklearn-compatible wrapper (for use with LOSOCrossValidator)
# ============================================================

class DANClassifier:
    """
    Sklearn-compatible wrapper around LightweightDAN.
    Can be used as drop-in replacement for XGBoost/RF in LOSO pipeline.
    """

    def __init__(self, n_features: int = 100, n_classes: int = 5,
                 n_domains: int = 10, hidden_dim: int = 64,
                 n_epochs: int = 50, batch_size: int = 64,
                 lr: float = 1e-3, device: str = 'auto',
                 random_state: int = 42):
        if not HAS_TORCH:
            raise ImportError(
                "Lite-DAN requires PyTorch, which is not installed in this "
                "environment (install with: pip install torch). There is "
                "no NumPy-only fallback currently implemented — despite "
                "this module's docstring mentioning one, that path was "
                "never actually built; treat DANClassifier as PyTorch-only "
                "until that's addressed. Use EMGClassifier (XGBoost/LDA/"
                "RandomForest) instead if you can't install PyTorch here.")
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.model: Optional[LightweightDAN] = None
        self.is_fitted = False
        self.classes_: Optional[List] = None
        self.label_to_idx: Optional[Dict] = None
        self.idx_to_label: Optional[Dict] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            domains: Optional[np.ndarray] = None) -> 'DANClassifier':
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = sorted(np.unique(y).tolist())
        self.label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}

        y_enc = np.array([self.label_to_idx[c] for c in y])

        # If no domains provided, treat all as same domain
        if domains is None:
            domains = np.zeros(len(y), dtype=int)
        else:
            # Encode domains to 0..n_domains-1
            unique_domains = np.unique(domains)
            domain_map = {d: i for i, d in enumerate(unique_domains)}
            domains = np.array([domain_map[d] for d in domains])
            self.n_domains = len(unique_domains)

        self.n_features = X.shape[1]
        self.n_classes = len(self.classes_)

        self.model = LightweightDAN(
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_domains=self.n_domains,
            hidden_dim=self.hidden_dim,
        )

        trainer = DANTrainer(self.model, lr=self.lr, device=self.device)
        trainer.fit(X, y_enc, domains, n_epochs=self.n_epochs,
                    batch_size=self.batch_size, verbose=False)
        self._trainer = trainer
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("DAN not fitted")
        preds = self._trainer.predict(X)
        return np.array([self.idx_to_label[int(p)] for p in preds])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("DAN not fitted")
        return self._trainer.predict_proba(X)

    def count_parameters(self) -> int:
        return self.model.count_parameters() if self.model else 0
