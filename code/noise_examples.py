"""Coherent over-rotation and readout-error channels."""
import numpy as np
from channels import alpha_beta, dag
from shadow_channel import noisy_global_shadow_channel, analytic_depol_form

def coherent_channel(V):
    """Coherent over-rotation channel."""
    return [np.asarray(V, dtype=complex)]

def readout_channel(R):
    """Readout-error channel from a confusion matrix."""
    d = R.shape[0]
    e = np.eye(d)
    return [np.sqrt(R[bp, b]) * np.outer(e[bp], e[b]) for b in range(d) for bp in range(d)]

def beta_coherent(V):
    """beta = sum |V_bb|^2 for coherent noise."""
    return float(np.sum(np.abs(np.diag(V)) ** 2))

def beta_readout(R):
    """beta = tr R for readout noise."""
    return float(np.trace(R))

def _channel_matches_depol(Ks, d, rng, ntest=6):
    _, be = alpha_beta(Ks, d)
    worst = 0.0
    for _ in range(ntest):
        G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        A = (G + dag(G)) / 2
        M = noisy_global_shadow_channel(A, Ks, d)
        P, _ = analytic_depol_form(A, be, d)
        worst = max(worst, np.linalg.norm(M - P) / max(np.linalg.norm(P), 1e-12))
    return worst
if __name__ == '__main__':
    import scipy.linalg as sla
    from scipy.stats import unitary_group
    rng = np.random.default_rng(0)
    d = 4
    V = unitary_group.rvs(d, random_state=rng)
    Ks = coherent_channel(V)
    a, b = alpha_beta(Ks, d)
    print('coherent  : alpha=%.3f beta=%.6f  vs sum|V_bb|^2=%.6f  channel-dev=%.1e' % (a, b, beta_coherent(V), _channel_matches_depol(Ks, d, rng)))
    H = unitary_group.rvs(d, random_state=rng)
    H = (H + dag(H)) / 2
    off2 = np.sum(np.abs(H - np.diag(np.diag(H))) ** 2)
    for eps in (0.05, 0.1, 0.2):
        _, be = alpha_beta(coherent_channel(sla.expm(1j * eps * H)), d)
        print('           eps=%.2f  d-beta=%.6f  eps^2||H_off||^2=%.6f' % (eps, d - be, eps ** 2 * off2))
    R = np.array([[0.9, 0.04, 0.03, 0.02], [0.05, 0.92, 0.02, 0.01], [0.03, 0.02, 0.93, 0.05], [0.02, 0.02, 0.02, 0.92]])
    R = R / R.sum(0, keepdims=True)
    Ks = readout_channel(R)
    a, b = alpha_beta(Ks, d)
    print('readout   : alpha=%.3f beta=%.6f  vs Tr R=%.6f  channel-dev=%.1e' % (a, b, beta_readout(R), _channel_matches_depol(Ks, d, rng)))
