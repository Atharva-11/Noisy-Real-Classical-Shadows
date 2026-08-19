"""Exact O(d) twirl at k = 2 and k = 3."""
import numpy as np
from numpy.linalg import pinv
from scipy.stats import ortho_group
from weingarten_core import basis_k3, perm_operator, omega_operator, hs

def basis_k2(d):
    """The commutant basis at k = 2: identity, swap, Omega."""
    I2 = np.eye(d * d)
    S = perm_operator((1, 0), d, k=2)
    Omega = np.zeros((d * d, d * d))
    vecOmega = np.zeros(d * d)
    for i in range(d):
        vecOmega[i * d + i] = 1.0
    Omega = np.outer(vecOmega, vecOmega)
    return ([I2, S, Omega], ['I', 'S', 'Omega'])

def twirl(X, d, k):
    """Exact O(d) twirl: the projector onto the commutant, Eq. (wein)."""
    if k == 2:
        ops, _ = basis_k2(d)
    elif k == 3:
        ops, _ = basis_k3(d)
    else:
        raise ValueError
    n = len(ops)
    G = np.array([[np.real(hs(ops[i], ops[j])) for j in range(n)] for i in range(n)])
    c = np.array([hs(ops[i], X) for i in range(n)])
    coeff = pinv(G) @ c
    return (sum((coeff[i] * ops[i] for i in range(n))), coeff, ops)

def twirl_MC(X, d, k, nsamp=20000, seed=1):
    """The same twirl by sampling, as a cross-check."""
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(X, dtype=float)
    for _ in range(nsamp):
        O = ortho_group.rvs(d, random_state=rng)
        Ok = O
        for _ in range(k - 1):
            Ok = np.kron(Ok, O)
        acc += Ok @ X @ Ok.T
    return acc / nsamp
if __name__ == '__main__':
    d = 3
    D2 = d * d
    rng = np.random.default_rng(0)
    X = rng.standard_normal((D2, D2))
    Texact, _, _ = twirl(X, d, 2)
    Tmc = twirl_MC(X, d, 2, nsamp=40000)
    print(f'k=2, d={d}: ||twirl_exact - twirl_MC||_F / ||exact|| = {np.linalg.norm(Texact - Tmc) / np.linalg.norm(Texact):.4f}')
    D3 = d ** 3
    X3 = rng.standard_normal((D3, D3))
    T3, _, _ = twirl(X3, d, 3)
    T3mc = twirl_MC(X3, d, 3, nsamp=40000)
    print(f'k=3, d={d}: ||twirl_exact - twirl_MC||_F / ||exact|| = {np.linalg.norm(T3 - T3mc) / np.linalg.norm(T3):.4f}')
    TT, _, _ = twirl(T3, d, 3)
    print('idempotent (k=3):', np.allclose(TT, T3, atol=1e-09))
