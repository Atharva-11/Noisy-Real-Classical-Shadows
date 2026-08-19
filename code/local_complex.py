"""Single-qubit complex-basis seminorm factors, at d = 2."""
import numpy as np
from scipy.stats import ortho_group, unitary_group
from channels import amplitude_damping_1q, depolarizing, apply_channel, adjoint_channel, dag

def single_qubit_complex_channel_superop(Ks1, W):
    """The d = 2 shadow channel as a superoperator."""
    from twirl_engine import twirl
    Estar = adjoint_channel(Ks1)

    def ptrace1(M):
        return np.einsum('aoai->oi', M.reshape(2, 2, 2, 2))

    def chan(A):
        out = np.zeros((2, 2), dtype=complex)
        AI = np.kron(A, np.eye(2))
        for k in range(2):
            w = W[:, k:k + 1]
            Pw = w @ w.conj().T
            EPw = apply_channel(Estar, Pw)
            T2, _, _ = twirl(np.kron(EPw, Pw), 2, 2)
            out += ptrace1(AI @ T2)
        return out
    basis = [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])]
    S = np.zeros((4, 4), dtype=complex)
    for j, E in enumerate(basis):
        S[:, j] = chan(E.astype(complex)).reshape(4)
    return S

def local_channel_MC(A, Ks1, Ws, nsamp, rng):
    """The same channel, by sampling."""
    from channels import tensor_channel
    n = len(Ws)
    d = 2 ** n
    Kfull = tensor_channel([Ks1] * n)
    acc = np.zeros((d, d), dtype=complex)
    for _ in range(nsamp):
        Us = [ortho_group.rvs(2, random_state=rng) for _ in range(n)]
        U = Us[0]
        for k in range(1, n):
            U = np.kron(U, Us[k])
        EUAU = apply_channel(Kfull, U @ A @ U.T)
        for wbits in range(d):
            bits = [wbits >> n - 1 - q & 1 for q in range(n)]
            wv = Ws[0][:, bits[0]:bits[0] + 1]
            for q in range(1, n):
                wv = np.kron(wv, Ws[q][:, bits[q]:bits[q] + 1])
            Pw = wv @ wv.conj().T
            amp = np.real(np.trace(EUAU @ Pw))
            acc += amp * (U.T @ Pw @ U)
    return acc / nsamp

def apply_superop_tensor(S1, A, n):
    """Apply a product of single-qubit superoperators."""
    d = 2 ** n
    T = A.reshape([2] * (2 * n))
    perm = []
    for j in range(n):
        perm += [j, n + j]
    T = np.transpose(T, perm).reshape([4] * n)
    for j in range(n):
        T = np.tensordot(S1, T, axes=([1], [j]))
        T = np.moveaxis(T, 0, j)
    T = T.reshape([2, 2] * n)
    inv = [0] * (2 * n)
    for j in range(n):
        inv[j] = 2 * j
        inv[n + j] = 2 * j + 1
    T = np.transpose(T, inv).reshape(2 ** n, 2 ** n)
    return T
if __name__ == '__main__':
    from scipy.stats import unitary_group
    rng = np.random.default_rng(21)
    print('=== Part IV: complex-basis LOCAL orthogonal ensemble ===\n')
    Ks1 = depolarizing(2, 0.85)
    W = unitary_group.rvs(2, random_state=rng)
    S1 = single_qubit_complex_channel_superop(Ks1, W)
    Estar = adjoint_channel(Ks1)
    beta = tbeta = 0.0
    for k in range(2):
        w = W[:, k:k + 1]
        Pw = w @ w.conj().T
        beta += np.real(np.trace(apply_channel(Ks1, Pw) @ Pw))
        tbeta += np.real(w.conj().T @ apply_channel(Ks1, Pw.T) @ w)[0, 0]
    f1 = (beta + tbeta - 2) / 4
    q = (3 * beta - 2 - tbeta) / (2 * (beta + tbeta - 2))

    def M1a(A):
        return f1 * (q * A + (1 - q) * A.T) + (1 - f1) * np.trace(A) * np.eye(2) / 2
    basis = [np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]]), np.array([[0, 0], [0, 1]])]
    S1an = np.zeros((4, 4), dtype=complex)
    for j, E in enumerate(basis):
        S1an[:, j] = M1a(E.astype(complex)).reshape(4)
    print(f'(a) single-qubit complex channel vs analytic (f1={f1:.4f}, q={q:.4f}): rel.err={np.linalg.norm(S1 - S1an) / np.linalg.norm(S1an):.2e}')
    A1 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    A2 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    lhs = apply_superop_tensor(S1, np.kron(A1, A2), 2)
    rhs = np.kron((S1 @ A1.reshape(4)).reshape(2, 2), (S1 @ A2.reshape(4)).reshape(2, 2))
    print(f'(b) factorization on product input: rel.err={np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs):.2e}')
    S1real = single_qubit_complex_channel_superop(Ks1, np.eye(2))

    def vdim(S1, n):
        d = 2 ** n
        M = np.zeros((d * d, d * d), dtype=complex)
        for col in range(d * d):
            E = np.zeros((d, d), dtype=complex)
            E[col // d, col % d] = 1
            M[:, col] = apply_superop_tensor(S1, E, n).reshape(d * d)
        return np.linalg.matrix_rank(M)
    print(f'(c) n=2 visible-space dim: complex={vdim(S1, 2)}/16 (full), real={vdim(S1real, 2)}/16 (=3^2, {{I,X,Z}} block)')
    Y = np.array([[0, -1j], [1j, 0]])
    rec = np.linalg.solve(S1, S1 @ Y.reshape(4)).reshape(2, 2)
    print(f'(d) recover Y (antisymmetric) complex basis: err={np.linalg.norm(rec - Y):.2e}; real-basis rank {np.linalg.matrix_rank(S1real)}/4 (Y in kernel)')
X1 = np.array([[0, 1], [1, 0]], complex)
Z1 = np.array([[1, 0], [0, -1]], complex)
Y1 = np.array([[0, -1j], [1j, 0]])

def single_qubit_invariants(Ks, W):
    """beta_1, betat_1, f_1 and q for one qubit."""
    Es = adjoint_channel(Ks)
    al = be = tb = ga = 0.0
    de = 0j
    ar = 0.0
    for k in range(2):
        w = W[:, k]
        ws = w.conj()
        Pw = np.outer(w, w.conj())
        A = apply_channel(Es, Pw)
        ov = np.sum(ws * ws)
        al += np.trace(A).real
        be += np.vdot(w, A @ w).real
        tb += np.vdot(ws, A @ ws).real
        ga += np.trace(A).real * abs(ov) ** 2
        de += ov * np.vdot(ws, A @ w)
        ar += abs(ov) ** 2
    return dict(alpha=al, beta=be, tbeta=tb, gamma=ga, delta=de, alpha_r=ar)

def c1_closed_form(Ks, W, P):
    """The single-qubit factor c_1, Eq. (c1_main)."""
    iv = single_qubit_invariants(Ks, W)
    f1 = (iv['beta'] + iv['tbeta'] - 2) / 4
    eta = 1 if np.allclose(P.T, P) else -1
    if eta == 1:
        return iv['alpha_r'] / (4 * f1 ** 2)
    twoq1 = (iv['beta'] - iv['tbeta']) / (2 * f1)
    return (2 - iv['alpha_r']) / (2 * f1 ** 2 * twoq1 ** 2)

def c1_exact(Ks, W, P):
    """The same factor, from the exact third moment."""
    from twirl_engine import twirl
    from exact_tools import shadow_superop
    S = shadow_superop(Ks, 2, W)
    Oh = (np.linalg.pinv(S).conj().T @ P.reshape(4)).reshape(2, 2)
    Es = adjoint_channel(Ks)
    R = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho = np.zeros((2, 2), complex)
            rho[i, j] = 1
            T3 = twirl(np.kron(np.kron(rho, Oh), Oh), 2, 3)[0]
            acc = 0j
            for k in range(2):
                w = W[:, k:k + 1]
                Pw = w @ w.conj().T
                acc += np.trace(T3 @ np.kron(np.kron(apply_channel(Es, Pw), Pw), Pw))
            R[j, i] = acc
    R = (R + R.conj().T) / 2
    ev = np.linalg.eigvalsh(R)
    return (float(ev.max()), float(ev.max() - ev.min()))

def verify_d2_complex(ntrial=4, seed=3):
    """Compare the two."""
    from scipy.stats import unitary_group
    rng = np.random.default_rng(seed)
    worst = 0.0
    chans = [('depol p=0.9', depolarizing(2, 0.9)), ('depol p=0.55', depolarizing(2, 0.55)), ('AD p=0.7', amplitude_damping_1q(0.7)), ('AD p=0.25', amplitude_damping_1q(0.25))]
    for _ in range(ntrial):
        Ks = [rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)) for _ in range(3)]
        S = sum((dag(K) @ K for K in Ks))
        L = np.linalg.cholesky(np.linalg.inv(S))
        chans.append(('random CPTP', [K @ L for K in Ks]))
    for nm, Ks in chans:
        W = unitary_group.rvs(2, random_state=rng)
        iv = single_qubit_invariants(Ks, W)
        aw = [abs(np.sum(W[:, k].conj() ** 2)) ** 2 for k in range(2)]
        worst = max(worst, abs(aw[0] - aw[1]))
        worst = max(worst, abs(2 * iv['delta'].real - (iv['beta'] + iv['tbeta'] - iv['alpha'] + iv['gamma'])))
        for P in (X1, Z1, Y1):
            ex, spread = c1_exact(Ks, W, P)
            cf = c1_closed_form(Ks, W, P)
            worst = max(worst, spread / max(abs(ex), 1e-300))
            worst = max(worst, abs(ex - cf) / max(abs(cf), 1e-300))
    for p in (0.9, 0.7, 0.4):
        W = unitary_group.rvs(2, random_state=rng)
        Ks = depolarizing(2, p)
        ar = single_qubit_invariants(Ks, W)['alpha_r']
        for P, cf in ((X1, 4 / (p * p * ar)), (Z1, 4 / (p * p * ar)), (Y1, 2 / (p * p * (2 - ar)))):
            ex, _ = c1_exact(Ks, W, P)
            worst = max(worst, abs(ex - cf) / cf)
    return worst

def complex_local_seminorm_factors(p=0.9, seed=11):
    """Per-qubit factors of Eq. (local_complex_seminorm)."""
    from channels import alpha_beta
    from scipy.stats import unitary_group
    Ks = depolarizing(2, p)
    _, beta = alpha_beta(Ks, 2)
    f1_real = (beta - 1) / 2
    Wc = unitary_group.rvs(2, random_state=np.random.default_rng(seed))
    iv = single_qubit_invariants(Ks, Wc)
    out = {'f1_real': f1_real, 'real_XZ': 1 / (2 * f1_real ** 2), 'alpha_r': iv['alpha_r'], 'f1': (iv['beta'] + iv['tbeta'] - 2) / 4}
    for lab, P in (('X', X1), ('Z', Z1), ('Y', Y1)):
        out['complex_' + lab] = c1_exact(Ks, Wc, P)[0]
        out['closed_' + lab] = c1_closed_form(Ks, Wc, P)
    return out
if __name__ == '__main__':
    print('\n=== Part IV seminorm factors: Prop. d2complex ===')
    r = complex_local_seminorm_factors()
    print(f'real basis: f_1={r['f1_real']:.3f}  c_1(X)=c_1(Z)=(2f_1^2)^-1={r['real_XZ']:.4f}  c_1(Y)=inf')
    print(f'complex basis (alpha_r={r['alpha_r']:.4f}, f_1={r['f1']:.4f}):')
    for lab in ('X', 'Z', 'Y'):
        print(f'   c_1({lab}) = {r['complex_' + lab]:.6f} (exact twirl)   {r['closed_' + lab]:.6f} (closed form)')
    print(f'worst relative error over all d=2 complex-basis checks: {verify_d2_complex():.2e}')
