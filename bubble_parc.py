import torch
from torch import nn
import numpy as np


def _gaussian_weights(centers, log_r, coords):
    """
    centers: (N, 3), log_r: (N,), coords: (G, 3) → W: (N, G)
    Uses ||a-b||² = ||a||² + ||b||² - 2a·b to avoid (N,G,3) intermediate.
    """
    r2  = log_r.exp().pow(2).unsqueeze(1)            # (N, 1)
    a2  = centers.pow(2).sum(1, keepdim=True)         # (N, 1)
    b2  = coords.pow(2).sum(1).unsqueeze(0)           # (1, G)
    ab  = centers @ coords.t()                        # (N, G)
    d2  = (a2 + b2 - 2 * ab).clamp(min=0)            # (N, G)
    m   = r2 * (-d2 / (2 * r2)).exp()                 # (N, G)  — r² scaling: larger bubbles exert more field
    return m / m.sum(1, keepdim=True).clamp(min=1e-8) # (N, G)


def _atlas_init(coords, labels, N):
    """
    Initialize N bubble centers from atlas parcel centroids.
    Radius = mean within-parcel distance from centroid.
    Extra bubbles (N > n_atlas) inherit the median radius and a random site.
    """
    coords = np.asarray(coords, dtype=np.float32)
    unique = sorted(int(l) for l in np.unique(labels) if l > 0)
    C     = np.zeros((N, 3), dtype=np.float32)
    log_r = np.zeros(N,      dtype=np.float32)

    for i in range(N):
        if i < len(unique):
            pts     = coords[labels == unique[i]]
            c       = pts.mean(0)
            r       = float(np.linalg.norm(pts - c, axis=1).mean())
        else:
            c = coords[np.random.randint(len(coords))]
            r = float(np.exp(np.median(log_r[:i]))) if i > 0 else 1.0
        C[i]     = c
        log_r[i] = np.log(max(r, 1.0))

    return torch.tensor(C), torch.tensor(log_r)


class BubbleParcellation(nn.Module):
    """
    Gaussian bubble pooling over CIFTI grayordinates.

    Cortical bubbles operate in fsLR sphere space (Euclidean ≈ geodesic).
    Subcortical bubbles operate in MNI mm space.
    Outputs Z ∈ R^{B×T×(N_c+N_s)} — first N_c tokens cortical, last N_s subcortical.

    Learnable parameters: C_c, log_r_c, log_a_c, C_s, log_r_s, log_a_s.
    Fixed buffers: coords_c (G_c, 3), coords_s (G_s, 3).
    """

    def __init__(self, coords_c, coords_s, labels, N_c, N_s):
        """
        coords_c: (G_c, 3) sphere coords for cortical grayordinates
        coords_s: (G_s, 3) MNI coords for subcortical grayordinates
        labels:   (G_c + G_s,) atlas parcel labels, 1-indexed (0 = medial wall / unlabeled)
        N_c, N_s: number of cortical / subcortical bubbles
        """
        super().__init__()
        self.G_c, self.G_s = len(coords_c), len(coords_s)
        self.N_c, self.N_s = N_c, N_s

        self.register_buffer('coords_c', torch.tensor(coords_c, dtype=torch.float32))
        self.register_buffer('coords_s', torch.tensor(coords_s, dtype=torch.float32))

        C_c, log_r_c = _atlas_init(coords_c, labels[:self.G_c], N_c)
        C_s, log_r_s = _atlas_init(coords_s, labels[self.G_c:], N_s)

        self.C_c     = nn.Parameter(C_c)
        self.log_r_c = nn.Parameter(log_r_c)
        self.C_s     = nn.Parameter(C_s)
        self.log_r_s = nn.Parameter(log_r_s)

    def weights(self):
        """Returns (W_c, W_s): (N_c, G_c) and (N_s, G_s) normalized Gaussian weights."""
        W_c = _gaussian_weights(self.C_c, self.log_r_c, self.coords_c)
        W_s = _gaussian_weights(self.C_s, self.log_r_s, self.coords_s)
        return W_c, W_s

    def forward(self, x, W_c=None, W_s=None):
        """
        x: (B, T, G_c + G_s)
        W_c, W_s: pre-computed weights (reuse if already computed for loss)
        → Z: (B, T, N_c + N_s)
        """
        if W_c is None or W_s is None:
            W_c, W_s = self.weights()
        Z_c = x[:, :, :self.G_c] @ W_c.t()
        Z_s = x[:, :, self.G_c:] @ W_s.t()
        return torch.cat([Z_c, Z_s], dim=-1)


def bubble_regularizers(W_c, W_s, log_r_c, log_r_s):
    """
    Returns (R_coverage, R_overlap, R_radius).
    All are non-negative; multiply by lambda and add to loss.

    R_coverage: 1 - mean max coverage per grayordinate (0 = every site fully owned)
    R_overlap:  mean pairwise bubble dot-product (0 = disjoint)
    R_radius:   mean squared radius (penalizes spread)
    """
    def _coverage(W):
        return 1.0 - W.max(0).values.mean()

    def _overlap(W):
        N = W.shape[0]
        if N < 2:
            return W.new_zeros(())
        ov = W @ W.t()                                 # (N, N)
        return (ov.sum() - ov.trace()) / (N * (N - 1))

    def _radius(log_r):
        return log_r.exp().pow(2).mean()

    R_cov     = (_coverage(W_c)   + _coverage(W_s))   / 2
    R_overlap = (_overlap(W_c)    + _overlap(W_s))    / 2
    R_radius  = (_radius(log_r_c) + _radius(log_r_s)) / 2
    return R_cov, R_overlap, R_radius

