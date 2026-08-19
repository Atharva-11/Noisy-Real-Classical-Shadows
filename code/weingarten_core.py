"""Brauer operators, the Gram matrix G_3(d), and Weingarten."""
import numpy as np
import itertools

def _ravel(idx, d):
    r = 0
    for x in idx:
        r = r * d + x
    return r

def perm_operator(perm, d, k=3):
    """Permutation operator S_pi."""
    perm = tuple(perm)
    inv = [0] * k
    for r in range(k):
        inv[perm[r]] = r
    D = d ** k
    M = np.zeros((D, D))
    for i in itertools.product(range(d), repeat=k):
        j = tuple((i[inv[r]] for r in range(k)))
        M[_ravel(j, d), _ravel(i, d)] = 1.0
    return M

def omega_operator(out_pair, in_pair, d, k=3):
    """Cup-cap operator Omega_{ab;xy}."""
    a, b = out_pair
    x, y = in_pair
    e = ({0, 1, 2} - {a, b}).pop()
    f = ({0, 1, 2} - {x, y}).pop()
    D = d ** k
    M = np.zeros((D, D))
    for out in itertools.product(range(d), repeat=k):
        if out[a] != out[b]:
            continue
        for inn in itertools.product(range(d), repeat=k):
            if inn[x] != inn[y]:
                continue
            if out[e] != inn[f]:
                continue
            M[_ravel(out, d), _ravel(inn, d)] = 1.0
    return M
S3 = {'e': (0, 1, 2), '(23)': (0, 2, 1), '(12)': (1, 0, 2), '(13)': (2, 1, 0), '(132)': (2, 0, 1), '(123)': (1, 2, 0)}
PERM_ORDER = ['e', '(23)', '(12)', '(13)', '(132)', '(123)']
OMEGA_ORDER = [((0, 1), (0, 1)), ((1, 2), (1, 2)), ((0, 2), (0, 2)), ((0, 1), (1, 2)), ((1, 2), (0, 1)), ((0, 2), (1, 2)), ((0, 2), (0, 1)), ((1, 2), (0, 2)), ((0, 1), (0, 2))]
OMEGA_LABELS = ['12;12', '23;23', '13;13', '12;23', '23;12', '13;23', '13;12', '23;13', '12;13']

def basis_k3(d):
    """The fifteen Brauer operators, in the order of Eq. (basisorder)."""
    ops = []
    labels = []
    for name in PERM_ORDER:
        ops.append(perm_operator(S3[name], d, 3))
        labels.append('S_' + name)
    for op_pair, lab in zip(OMEGA_ORDER, OMEGA_LABELS):
        ops.append(omega_operator(op_pair[0], op_pair[1], d, 3))
        labels.append('Om_' + lab)
    return (ops, labels)

def hs(A, B):
    """Hilbert-Schmidt inner product tr[A^dag B]."""
    return np.tensordot(A.conj(), B, axes=([0, 1], [0, 1])).item()

def gram_k3(d):
    """The order-3 Gram matrix G_3(d) = (d^ell(i,j))."""
    ops, labels = basis_k3(d)
    n = len(ops)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = np.real(hs(ops[i], ops[j]))
    return (G, labels)
if __name__ == '__main__':
    for d in [3, 4, 5]:
        G, labels = gram_k3(d)
        ell = np.round(np.log(G) / np.log(d)).astype(int)
        print(f'd={d}: rank(G)={np.linalg.matrix_rank(G)}  det={np.linalg.det(G):.3e}')
    G2, labels = gram_k3(2)
    print('\nd=2: rank(G) =', np.linalg.matrix_rank(G2), ' (expect 10, nullity 5)')
    print('labels order:', labels)
