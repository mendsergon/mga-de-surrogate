"""
Neural Network Surrogate Model

A multilayer perceptron that learns to approximate an expensive fitness
function from a dataset of (decision_vector, fitness) pairs.

Used to pre-screen candidate solutions during DE to avoid wasting
expensive Lambert solves on clearly-bad candidates.

Jin, Y. (2011). Surrogate-assisted evolutionary computation: Recent
advances and future challenges. Swarm and Evolutionary Computation,
1(2), 61-70.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class Surrogate:
    """
    MLP surrogate for expensive fitness functions.

    Fitness values are clipped to a cap before training — extreme
    penalty values (1e6 from infeasible solutions) would destroy
    training gradients otherwise.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        hidden_layers: Tuple[int, ...] = (128, 128, 64),
        max_iter: int = 500,
        fitness_cap: float = 100.0,
        random_state: int = 0,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.fitness_cap = fitness_cap
        self.random_state = random_state

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
            learning_rate_init=1e-3,
        )
        self.y_scaler = StandardScaler()
        self.fitted = False
        self.train_size = 0

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        """Map decision vectors to [0, 1] using problem bounds."""
        return (X - self.lb) / (self.ub - self.lb)

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        """Clip extreme penalty values before training."""
        return np.clip(y, None, self.fitness_cap)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the surrogate on a dataset of (x, fitness) pairs."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if len(X) < 20:
            # Too few points — don't train
            self.fitted = False
            return

        X_norm = self._normalize_x(X)
        y_clipped = self._prepare_y(y)
        y_scaled = self.y_scaler.fit_transform(y_clipped.reshape(-1, 1)).ravel()

        self.model.fit(X_norm, y_scaled)
        self.fitted = True
        self.train_size = len(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fitness for new candidates."""
        if not self.fitted:
            raise RuntimeError("Surrogate is not fitted yet.")

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X_norm = self._normalize_x(X)
        y_scaled = self.model.predict(X_norm)
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate prediction quality on held-out data."""
        if not self.fitted:
            return {"mae": np.nan, "rmse": np.nan, "rank_corr": np.nan}

        y_pred = self.predict(X)
        y_true = np.clip(np.asarray(y), None, self.fitness_cap)

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))

        # Spearman rank correlation — how well does the surrogate
        # rank candidates, even if absolute values are off
        rank_true = np.argsort(np.argsort(y_true))
        rank_pred = np.argsort(np.argsort(y_pred))
        rank_corr = float(np.corrcoef(rank_true, rank_pred)[0, 1])

        return {"mae": mae, "rmse": rmse, "rank_corr": rank_corr}


class DataCollector:
    """
    Accumulates (x, f) pairs observed during optimization.

    Used as the training reservoir for the surrogate model.
    Capped to max_size entries with FIFO eviction.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    def add(self, x: np.ndarray, f: float) -> None:
        self.X.append(np.copy(x))
        self.y.append(float(f))
        if len(self.X) > self.max_size:
            self.X.pop(0)
            self.y.pop(0)

    def add_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        for xi, fi in zip(X, y):
            self.add(xi, fi)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.X), np.array(self.y)

    def __len__(self) -> int:
        return len(self.X)
