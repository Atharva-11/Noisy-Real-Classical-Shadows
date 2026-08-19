"""Exact mean and variance, via exact Haar-O(d) twirls."""
import numpy as np
from twirl_engine import twirl
from channels import apply_channel, adjoint_channel, alpha_beta, dag

def basis_projectors(W):
    """Projectors Pi_w of a measurement basis."""
    d = W.shape[0]
    return [W[:, k:k + 1] @ W[:, k:k + 1].conj().T for k in range(d)]

def shadow_superop(Ks, d, W):
    """The shadow channel as a d^2 x d^2 superoperator."""
    Estar = adjoint_channel(Ks)
    Pw = basis_projectors(W)

    def ptrace1(M):
        return np.einsum('aoai->oi', M.reshape(d, d, d, d))
    S = np.zeros((d * d, d * d), dtype=complex)
    for col in range(d * d):
        A = np.zeros((d, d), dtype=complex)
        A[col // d, col % d] = 1
        out = np.zeros((d, d), dtype=complex)
        AI = np.kron(A, np.eye(d))
        for P in Pw:
            EP = apply_channel(Estar, P)
            T2, _, _ = twirl(np.kron(EP, P), d, 2)
            out += ptrace1(AI @ T2)
        S[:, col] = out.reshape(d * d)
    return S

def exact_mean_var(O, rho, Ks, d, W):
    """Exact E[o] and Var[o], with no sampling."""
    S = shadow_superop(Ks, d, W)
    Sinv = np.linalg.pinv(S)
    Ohat = (Sinv.conj().T @ O.reshape(d * d)).reshape(d, d)
    Estar = adjoint_channel(Ks)
    Pw = basis_projectors(W)
    T3 = twirl(np.kron(np.kron(rho, Ohat), Ohat), d, 3)[0]
    T2 = twirl(np.kron(rho, Ohat), d, 2)[0]
    Eo2 = 0.0
    Eo = 0.0
    for P in Pw:
        EP = apply_channel(Estar, P)
        Eo2 += np.real(np.trace(T3 @ np.kron(np.kron(EP, P), P)))
        Eo += np.real(np.trace(T2 @ np.kron(EP, P)))
    return (Eo, Eo2 - Eo ** 2)
if __name__ == '__main__':
    from channels import depolarizing
    from scipy.stats import unitary_group
    rng = np.random.default_rng(0)
    d = 4
    Ks = depolarizing(d, 0.85)
    _, beta = alpha_beta(Ks, d)
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = G @ dag(G)
    rho /= np.trace(rho)
    M = rng.standard_normal((d, d))
    O = (M + M.T) / 2

    def var_O(A0, rho, beta, d):
        t1 = np.real(np.trace(A0 @ A0))
        t2 = np.real(np.trace(rho @ A0 @ A0))
        t3 = np.real(np.trace(A0 @ rho))
        return (d - 1) * (d + 2) / ((d + 4) * (beta - 1)) * ((d * (d + 3) - 4 * beta) / (2 * d * (beta - 1)) * t1 + 2 * t2) - t3 ** 2
    O0 = O - np.trace(O) * np.eye(d) / d
    Eo, V = exact_mean_var(O, rho, Ks, d, np.eye(d))
    print('real basis: exact Var =', round(V, 6), ' analytic var_O =', round(var_O(O0, rho, beta, d), 6), ' mean ok:', np.isclose(Eo, np.real(np.trace(O0 @ rho))))
