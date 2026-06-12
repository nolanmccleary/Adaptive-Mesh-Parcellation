import torch
from torch import nn
import numpy as np


class SoftParcellation(nn.Module):
    """
    Learnable soft pooling: Z = X @ softmax(A, dim=1).T
      A: (N, G) learned parameter
      N: number of bubbles (hyperparameter)
      G: total grayordinates (91282)

    Atlas init: A[i, g] = 5.0 if labels[g] == i+1 else -10.0
    """

    def __init__(self, G: int, N: int, labels: np.ndarray):
        super().__init__()
        A = torch.full((N, G), -10.0)
        n_atlas = int(labels.max())
        for i in range(min(N, n_atlas)):
            A[i, labels == (i + 1)] = 5.0
        self.A = nn.Parameter(A)
        self.G = G
        self.N = N

    def weights(self):
        return torch.softmax(self.A, dim=1)   # (N, G)

    def forward(self, x):                     # x: (B, T, G)
        return x @ self.weights().t()         # (B, T, N)
