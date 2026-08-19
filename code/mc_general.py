"""Monte-Carlo shadows for a general channel and measurement basis."""
import numpy as np

def _haar_orthogonal(batch, d, rng):
    G = rng.standard_normal((batch, d, d))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]

def run(rho, Ohat, W, d, p, N, rng, batch=None):
    """Run the protocol for a general channel and basis."""
    if batch is None:
        batch = min(20000, max(32, int(4200000.0 / (d * d))))
    out = np.empty(N)
    done = 0
    while done < N:
        nb = min(batch, N - done)
        U = _haar_orthogonal(nb, d, rng)
        Y = np.matmul(U.transpose(0, 2, 1), W)
        Yc = Y.conj()
        pr = (Yc * np.matmul(rho, Y)).sum(1).real
        pr = p * pr + (1.0 - p) / d
        np.clip(pr, 0, None, out=pr)
        pr /= pr.sum(1, keepdims=True)
        k = (rng.random(nb)[:, None] < np.cumsum(pr, 1)).argmax(1)
        OY = np.matmul(Ohat, Y)
        idx = k[:, None, None]
        ycol = np.take_along_axis(Yc, idx, axis=2)[:, :, 0]
        ocol = np.take_along_axis(OY, idx, axis=2)[:, :, 0]
        out[done:done + nb] = (ycol * ocol).sum(1).real
        done += nb
    return out

def mean_ci(samples, z=1.96):
    """Mean and 95% confidence interval."""
    m = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(samples.size)
    return (m, z * se)

def var_ci(samples, z=1.96):
    """Variance and 95% confidence interval."""
    n = samples.size
    v = samples.var(ddof=1)
    mu4 = ((samples - samples.mean()) ** 4).mean()
    se = np.sqrt(max(mu4 - v * v, 0.0) / n)
    return (v, z * se)
