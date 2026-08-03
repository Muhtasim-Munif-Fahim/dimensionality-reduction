"""Autoencoder implemented from scratch with numpy (single hidden layer)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class NumpyAutoencoder:
    def __init__(self, n_input: int, n_hidden: int, learning_rate: float = 0.01):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.W1 = np.random.randn(n_input, n_hidden) * 0.01
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_input) * 0.01
        self.b2 = np.zeros((1, n_input))

    def encode(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.W1 + self.b1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = self.encode(X)
        return h @ self.W2 + self.b2

    def fit(self, X: np.ndarray, epochs: int = 100, verbose: bool = False) -> list[float]:
        losses = []
        for epoch in range(epochs):
            h = self.encode(X)
            output = h @ self.W2 + self.b2
            error = output - X
            loss = float(np.mean(error**2))
            losses.append(loss)

            dW2 = h.T @ error
            db2 = error.sum(axis=0, keepdims=True)
            dh = error @ self.W2.T * h * (1 - h)
            dW1 = X.T @ dh
            db1 = dh.sum(axis=0, keepdims=True)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"epoch {epoch+1}: loss={loss:.4f}")
        return losses

    def reconstruct_error(self, X: np.ndarray) -> float:
        output = self.forward(X)
        return float(np.mean((output - X) ** 2))
