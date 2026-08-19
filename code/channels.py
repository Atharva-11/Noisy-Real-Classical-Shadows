"""Noise channels, and the scalar invariants alpha and beta."""
import numpy as np
import itertools

def dag(K):
    """Conjugate transpose."""
    return K.conj().T

def depolarizing(d, p):
    """Kraus operators of the depolarizing channel, Eq. (depol_def)."""
    Ks = [np.sqrt(p) * np.eye(d)]
    om = np.exp(2j * np.pi / d)
    X = np.roll(np.eye(d), 1, axis=0)
    Z = np.diag([om ** k for k in range(d)])
    coeff = np.sqrt((1 - p) / (d * d))
    for a in range(d):
        for b in range(d):
            Ks.append(coeff * np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b))
    return Ks

def amplitude_damping_1q(p):
    """Single-qubit amplitude damping."""
    K0 = np.array([[1, 0], [0, np.sqrt(p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(1 - p)], [0, 0]], dtype=complex)
    return [K0, K1]

def tensor_channel(Ks_list):
    """Kraus operators of a tensor product of channels."""
    out = [np.array([[1]], dtype=complex)]
    for Ks in Ks_list:
        out = [np.kron(A, K) for A in out for K in Ks]
    return out

def apply_channel(Ks, rho):
    """Apply a channel given by Kraus operators."""
    return sum((K @ rho @ dag(K) for K in Ks))

def adjoint_channel(Ks):
    """Dual channel: Kraus operators K_i^dag."""
    return [dag(K) for K in Ks]

def diag_channel(A):
    """The completely dephasing channel diag."""
    d = A.shape[0]
    return np.diag(np.diag(A))

def alpha_beta(Ks, d):
    """alpha = tr[E(1)] and beta = tr[E o diag], Eq. (scalars)."""
    I = np.eye(d)
    E_I = apply_channel(Ks, I)
    alpha = np.real(np.trace(E_I))
    beta = 0.0
    for b in range(d):
        Pb = np.zeros((d, d))
        Pb[b, b] = 1
        beta += np.real(np.trace(apply_channel(Ks, Pb) @ Pb))
    return (alpha, beta)
if __name__ == '__main__':
    for d, p in [(4, 0.8), (8, 0.9)]:
        Ks = depolarizing(d, p)
        S = sum((dag(K) @ K for K in Ks))
        a, b = alpha_beta(Ks, d)
        print(f'depol d={d} p={p}: TP={np.allclose(S, np.eye(d))}, alpha={a:.4f} (=d? {np.isclose(a, d)}), beta={b:.4f}, pd+1-p={p * d + 1 - p:.4f}')
    for n, p in [(2, 0.7), (3, 0.5)]:
        d = 2 ** n
        Ks = tensor_channel([amplitude_damping_1q(p)] * n)
        a, b = alpha_beta(Ks, d)
        print(f'AD n={n} p={p}: alpha={a:.4f}(=d {np.isclose(a, d)}), beta={b:.4f}, (1+p)^n={(1 + p) ** n:.4f}')
