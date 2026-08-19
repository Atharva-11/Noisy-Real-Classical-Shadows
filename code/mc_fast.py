"""Vectorised inner loop shared by the Monte-Carlo drivers."""
import numpy as np
from channels import apply_channel

def _haar_orthogonal(batch, d, rng):
    G = rng.standard_normal((batch, d, d))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]

def _channel_diag_batch(Ks, Xb):
    """Outcome probabilities for a batch of unitaries."""
    out = np.zeros((Xb.shape[0], Xb.shape[1]), dtype=complex)
    for K in Ks:
        Y = np.einsum('xy,iyz,wz->ixw', K, Xb, np.conj(K), optimize=True)
        out += np.diagonal(Y, axis1=1, axis2=2)
    return np.real(out)

def run_trajectories(rho, Oobs, Ks, d, beta, N, M, rng, batch=None):
    """Sample shadow estimates."""
    if batch is None:
        batch = min(50000, max(32, int(4200000.0 / (d * d))))
    f = 2 * (beta - 1) / ((d - 1) * (d + 2))
    Ohat = 1 / f * Oobs + (1 - 1 / f) * np.trace(Oobs) / d * np.eye(d)
    ests = np.empty((M, N))
    for m in range(M):
        done = 0
        while done < N:
            b = min(batch, N - done)
            Q = _haar_orthogonal(b, d, rng)
            X = np.einsum('ixy,yz,iwz->ixw', Q, rho, Q, optimize=True)
            probs = _channel_diag_batch(Ks, X)
            probs = np.clip(probs, 0, None)
            probs /= probs.sum(1, keepdims=True)
            u = rng.random(b)
            cs = np.cumsum(probs, 1)
            bo = (u[:, None] < cs).argmax(1)
            o = Q[np.arange(b), bo, :]
            ests[m, done:done + b] = np.real(np.einsum('ix,xy,iy->i', o, Ohat, o, optimize=True))
            done += b
    return ests
