"""An orthogonal 2-design that is not a 3-design, and beats one. Sec. 8."""
import numpy as np
__all__ = ['G288_generators', 'close_group', 'dim_comm', 'real_clifford_2', 'shadow_channel_matrix', 'second_moment', 'report']
_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
_Z = np.diag([1.0, -1.0])
_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_I = np.eye(2)
_CZ = np.diag([1.0, 1.0, 1.0, -1.0])
_CX = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
_SW = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])

def G288_generators():
    """The three generators of Gamma_288, Eq. (g288)."""
    h = 0.5
    A = np.array([[h, -h, -h, -h], [h, -h, h, h], [-h, -h, h, -h], [-h, -h, -h, h]])
    B = np.array([[0.0, 0.0, 0.0, -1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
    C = np.array([[h, -h, h, h], [h, -h, -h, -h], [h, h, h, -h], [-h, -h, h, -h]])
    return [A, B, C]

def _canon(A, tol=1e-09):
    B = A.copy()
    i = int(np.argmax(np.abs(B) > tol))
    if B.flat[i] < 0:
        B = -B
    return tuple(np.round(B.flat, 6))

def close_group(gens, cap=100000):
    """Close a set of generators into the finite group they generate."""
    d = gens[0].shape[0]
    E = np.eye(d)
    seen = {_canon(E): E}
    frontier = [E]
    while frontier:
        nxt = []
        for A in frontier:
            for g in gens:
                B = g @ A
                k = _canon(B)
                if k not in seen:
                    seen[k] = B
                    nxt.append(B)
                    if len(seen) > cap:
                        raise RuntimeError('group exceeded cap %d' % cap)
        frontier = nxt
    return list(seen.values())

def dim_comm(G, k):
    """dim Comm(G,k) = |G|^-1 sum_g (tr U_g)^{2k}, Eq. (dimcomm_char)."""
    t = np.array([np.trace(U) for U in G], dtype=float)
    return float(np.mean(np.abs(t) ** (2 * k)))

def real_clifford_2():
    """The real two-qubit Clifford group C_2 cap O(4), order 1152."""
    return close_group([np.kron(_H, _I), np.kron(_I, _H), np.kron(_Z, _I), np.kron(_I, _Z), np.kron(_X, _I), np.kron(_I, _X), _CZ, _CX, _SW])

def _depol_kraus(d, p):
    om = np.exp(2j * np.pi / d)
    X = np.roll(np.eye(d), 1, axis=0)
    Z = np.diag([om ** k for k in range(d)])
    c = np.sqrt((1 - p) / (d * d))
    return [np.sqrt(p) * np.eye(d)] + [c * np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b) for a in range(d) for b in range(d)]

def shadow_channel_matrix(G, Ks, d):
    """The shadow channel of an ensemble, as a superoperator on vec(A)."""
    E = lambda A: sum((K @ A @ K.conj().T for K in Ks))
    Pi = [np.diag([1.0 if i == b else 0.0 for i in range(d)]) for b in range(d)]
    M = np.zeros((d * d, d * d), complex)
    for k in range(d * d):
        v = np.zeros(d * d, complex)
        v[k] = 1
        acc = np.zeros((d, d), complex)
        for U in G:
            EU = E(U @ v.reshape(d, d) @ U.T)
            for b in range(d):
                acc += np.real(np.trace(Pi[b] @ EU)) * (U.T @ Pi[b] @ U)
        M[:, k] = (acc / len(G)).reshape(-1)
    return M

def second_moment(G, Ks, Minv, O0, rho, d):
    """Single-shot E[o^2] for one ensemble."""
    E = lambda A: sum((K @ A @ K.conj().T for K in Ks))
    Pi = [np.diag([1.0 if i == b else 0.0 for i in range(d)]) for b in range(d)]
    Ohat = (Minv.conj().T @ O0.reshape(-1)).reshape(d, d)
    tot = 0.0
    for U in G:
        EU = E(U @ rho @ U.T)
        for b in range(d):
            tot += np.real(np.trace(Pi[b] @ EU)) * np.real(np.trace(Pi[b] @ (U @ Ohat @ U.T))) ** 2
    return tot / len(G)
WITNESS = np.array([[-1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
WITNESS_STATE = np.diag([0.55, 0.25, 0.15, 0.05])

def report(p=0.9, verbose=True):
    """The numbers Sec. 8 quotes: the commutant dimensions, the channel agreement at k <= 2, and the 20.6% second-moment gap at k = 3."""
    d = 4
    G = close_group(G288_generators())
    RC = real_clifford_2()
    Ks = _depol_kraus(d, p)
    M288 = shadow_channel_matrix(G, Ks, d)
    MRC = shadow_channel_matrix(RC, Ks, d)
    Minv = np.linalg.pinv(MRC, rcond=1e-10)
    O0 = WITNESS - np.trace(WITNESS) * np.eye(d) / d
    O0 = O0 / np.sqrt(np.trace(O0 @ O0))
    rho = WITNESS_STATE / np.trace(WITNESS_STATE)
    m288 = second_moment(G, Ks, Minv, O0, rho, d)
    mRC = second_moment(RC, Ks, Minv, O0, rho, d)
    P = np.kron(_Z, _Z)
    P = P / np.sqrt(np.trace(P @ P))
    pauli = (second_moment(G, Ks, Minv, P, rho, d), second_moment(RC, Ks, Minv, P, rho, d))
    out = {'order_G288': len(G), 'order_RC2': len(RC), 'dimC2_G288': dim_comm(G, 2), 'dimC3_G288': dim_comm(G, 3), 'dimC2_RC2': dim_comm(RC, 2), 'dimC3_RC2': dim_comm(RC, 3), 'channel_dev': float(np.abs(M288 - MRC).max()), 'm2_G288': m288, 'm2_RC2': mRC, 'ratio': m288 / mRC, 'pauli_ratio': pauli[0] / pauli[1], 'subgroup': all((any((np.abs(np.abs(A) - np.abs(B)).max() < 1e-09 for B in RC)) for A in G))}
    if verbose:
        print('G288 : order %d, dim Comm2 = %.4f, dim Comm3 = %.4f' % (out['order_G288'], out['dimC2_G288'], out['dimC3_G288']))
        print('RC_2 : order %d, dim Comm2 = %.4f, dim Comm3 = %.4f' % (out['order_RC2'], out['dimC2_RC2'], out['dimC3_RC2']))
        print('G288 <= RC_2                       : %s' % out['subgroup'])
        print('max |shadow channel difference|    : %.3e   (identical: a k<=2 statement)' % out['channel_dev'])
        print('E[o^2] G288 / E[o^2] 3-design      : %.6f   (%.2f%% lower)' % (out['ratio'], 100 * (1 - out['ratio'])))
        print('the same ratio for a Pauli string  : %.10f' % out['pauli_ratio'])
    return out
if __name__ == '__main__':
    report()
