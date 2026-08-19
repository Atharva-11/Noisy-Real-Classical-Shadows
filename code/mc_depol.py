"""Monte-Carlo orthogonal shadows under depolarizing noise."""
import numpy as np

def _haar_orthogonal(batch, d, rng):
    G = rng.standard_normal((batch, d, d))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]

def run_depol(rho, O0, d, p, beta, N, M, rng, batch=None):
    """Run the protocol under depolarizing noise."""
    if batch is None:
        batch = min(20000, max(32, int(8400000.0 / (d * d))))
    f = 2 * (beta - 1) / ((d - 1) * (d + 2))
    Ohat = 1 / f * O0
    out = np.empty((M, N))
    for m in range(M):
        done = 0
        while done < N:
            bsz = min(batch, N - done)
            Q = _haar_orthogonal(bsz, d, rng)
            Rq = Q @ rho
            diag = np.einsum('ixy,ixy->ix', Rq, Q, optimize=True)
            probs = p * diag + (1 - p) / d
            probs = np.clip(probs, 0, None)
            probs /= probs.sum(1, keepdims=True)
            u = rng.random(bsz)
            bo = (u[:, None] < np.cumsum(probs, 1)).argmax(1)
            o = Q[np.arange(bsz), bo, :]
            out[m, done:done + bsz] = np.einsum('ix,xy,iy->i', o, Ohat, o, optimize=True)
            done += bsz
    return out
