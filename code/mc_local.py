"""Monte-Carlo local (product-ensemble) shadows."""
import numpy as np
from channels import apply_channel, adjoint_channel
I2 = np.eye(2)
PX = np.array([[0.0, 1.0], [1.0, 0.0]])
PY = np.array([[0.0, -1j], [1j, 0.0]])
PZ = np.array([[1.0, 0.0], [0.0, -1.0]])
PAULI = {'I': I2, 'X': PX, 'Y': PY, 'Z': PZ}

def haar_O2(B, rng):
    """Haar-random element of O(2)."""
    th = rng.uniform(0, 2 * np.pi, B)
    c, s = (np.cos(th), np.sin(th))
    det = rng.integers(0, 2, B) * 2.0 - 1.0
    G = np.empty((B, 2, 2))
    G[:, 0, 0] = c
    G[:, 0, 1] = -s * det
    G[:, 1, 0] = s
    G[:, 1, 1] = c * det
    return G.astype(complex)

def haar_U2(B, rng):
    """Haar-random element of U(2)."""
    A = rng.standard_normal((B, 2, 2)) + 1j * rng.standard_normal((B, 2, 2))
    Q, R = np.linalg.qr(A)
    ph = np.diagonal(R, axis1=1, axis2=2)
    return Q * (ph / np.abs(ph))[:, None, :]

def classical_map(kind, s):
    """The single-qubit inverse applied to a snapshot."""
    if kind == 'depolarizing':
        return np.array([[s + (1 - s) / 2, (1 - s) / 2], [(1 - s) / 2, s + (1 - s) / 2]])
    if kind == 'dephasing':
        return np.eye(2)
    if kind == 'amplitude damping':
        return np.array([[1.0, 1.0 - s], [0.0, s]])
    if kind == 'readout':
        return np.array([[1.0 - s, s], [s, 1.0 - s]])
    raise ValueError(kind)

def check_classical_map(kind, s, rng, ntest=4):
    from noise_zoo import family
    Ks, _b, _l = family(kind, 1, s)
    K = classical_map(kind, s)
    worst = 0.0
    for _ in range(ntest):
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        sig = A @ A.conj().T
        sig /= np.trace(sig)
        lhs = np.real(np.diag(apply_channel(Ks, sig)))
        rhs = K @ np.real(np.diag(sig))
        worst = max(worst, np.max(np.abs(lhs - rhs)))
    return worst

def local_scalars(Ks1, W=None):
    """beta_1 and f_1 for one qubit."""
    if W is None:
        W = np.eye(2, dtype=complex)
    b = bt = 0.0
    for k in range(2):
        w = W[:, k:k + 1]
        Pw = w @ w.conj().T
        EP = apply_channel(Ks1, Pw)
        b += np.real(w.conj().T @ EP @ w)[0, 0]
        bt += np.real(np.conj(w).conj().T @ EP @ np.conj(w))[0, 0]
    return (float(b), float(bt))

def inverse_coeffs(Ks1, W=None, ensemble='O'):
    """Coefficients of the single-qubit inverse."""
    b1, bt1 = local_scalars(Ks1, W)
    if ensemble == 'U':
        fU = (b1 - 1) / 3.0
        return ({'X': 1 / fU, 'Y': 1 / fU, 'Z': 1 / fU}, dict(f1=fU, beta1=b1, betat1=bt1))
    f1 = (b1 + bt1 - 2) / 4.0
    q = (3 * b1 - 2 - bt1) / (2 * (b1 + bt1 - 2))
    cy = None if abs(2 * q - 1) < 1e-12 else 1.0 / (f1 * (2 * q - 1))
    return ({'X': 1 / f1, 'Y': cy, 'Z': 1 / f1}, dict(f1=f1, q=q, beta1=b1, betat1=bt1))

def _apply_1q(psi, G, q, n):
    B = psi.shape[0]
    T = psi.reshape(B, 2 ** q, 2, 2 ** (n - q - 1))
    return np.einsum('bij,bajc->baic', G, T, optimize=True).reshape(B, -1)

def run_local(psi, n, pauli, Ks1, noise_kind, s, N, rng, ensemble='O', W=None, extra_gate=None, batch=None):
    """Run the local protocol."""
    coeffs, info = inverse_coeffs(Ks1, W, ensemble)
    sites = [j for j, ch in enumerate(pauli) if ch != 'I']
    for j in sites:
        if coeffs[pauli[j]] is None:
            raise ValueError(f'{pauli[j]} is not in the visible space for this basis')
    K = classical_map(noise_kind, s)
    Wm = np.eye(2, dtype=complex) if W is None else np.asarray(W, dtype=complex)
    draw = haar_O2 if ensemble == 'O' else haar_U2
    if batch is None:
        batch = int(min(4000, max(200, 2000000.0 / 2 ** n)))
    out = np.empty(N)
    done = 0
    while done < N:
        B = min(batch, N - done)
        Us = [draw(B, rng) for _ in range(n)]
        st = np.tile(psi.astype(complex), (B, 1))
        for j in range(n):
            st = _apply_1q(st, Us[j], j, n)
        meas = st
        if extra_gate is not None:
            g = np.tile(np.asarray(extra_gate, dtype=complex), (B, 1, 1))
            for j in range(n):
                meas = _apply_1q(meas, g, j, n)
        if W is not None:
            wd = np.tile(Wm.conj().T, (B, 1, 1))
            for j in range(n):
                meas = _apply_1q(meas, wd, j, n)
        pr = np.abs(meas) ** 2
        pr /= pr.sum(1, keepdims=True)
        idx = (rng.random(B)[:, None] < np.cumsum(pr, 1)).argmax(1)
        bits = idx[:, None] >> np.arange(n - 1, -1, -1)[None, :] & 1
        u = rng.random((B, n))
        p1 = K[1, 0] * (bits == 0) + K[1, 1] * (bits == 1)
        bits = (u < p1).astype(int)
        est = np.ones(B)
        for j in sites:
            wcol = Wm[:, 0][None, :] * (bits[:, j] == 0)[:, None] + Wm[:, 1][None, :] * (bits[:, j] == 1)[:, None]
            v = np.einsum('bij,bi->bj', Us[j].conj(), wcol, optimize=True)
            Pm = PAULI[pauli[j]]
            val = np.einsum('bi,ij,bj->b', v.conj(), Pm, v, optimize=True)
            est = est * np.real(val) * coeffs[pauli[j]]
        out[done:done + B] = est
        done += B
    return (out, info)

def predict_second_moment(Ks1, pauli, W=None, ensemble='O'):
    """Predicted second moment, Eq. (pauli_seminorm)."""
    b1, bt1 = local_scalars(Ks1, W)
    k = sum((1 for c in pauli if c != 'I'))
    if ensemble == 'U':
        fU = (b1 - 1) / 3.0
        return (3 * fU ** 2) ** (-k)
    if W is None:
        f1 = (b1 - 1) / 2.0
        return (2 * f1 ** 2) ** (-k)
    from local_complex import c1_closed_form
    val = 1.0
    for ch in pauli:
        if ch != 'I':
            val *= c1_closed_form(Ks1, np.asarray(W, dtype=complex), PAULI[ch])
    return float(np.real(val))

def ci(samples, z=1.96):
    """95% confidence interval."""
    n = samples.size
    m = samples.mean()
    se = samples.std(ddof=1) / np.sqrt(n)
    v = samples.var(ddof=1)
    mu4 = ((samples - samples.mean()) ** 4).mean()
    vse = np.sqrt(max(mu4 - v * v, 0.0) / n)
    return (m, z * se, v, z * vse)

def moment2(samples, z=1.96):
    """Second moment and its confidence interval."""
    s2 = samples ** 2
    return (s2.mean(), z * s2.std(ddof=1) / np.sqrt(s2.size))

def pauli_matrix(pauli):
    """Pauli string as a matrix."""
    M = None
    for ch in pauli:
        M = PAULI[ch] if M is None else np.kron(M, PAULI[ch])
    return M

def _resid(val, pred, cival, tol=1e-12):
    if cival > tol:
        return '%+6.2f CI' % ((val - pred) / cival)
    return '%.1e rel' % (abs(val - pred) / max(abs(pred), 1e-15))
if __name__ == '__main__':
    from channels import depolarizing, amplitude_damping_1q
    from many_body import tfim, ground_state
    rng = np.random.default_rng(4)
    print('=== E1: end-to-end LOCAL orthogonal shadows (Parts II and IV) ===\n')
    print("classical diagonal-map identity <b|E(s)|b> = sum K_bb' s_b'b'  (max abs dev):")
    for kd, s in [('depolarizing', 0.9), ('dephasing', 0.6), ('amplitude damping', 0.7), ('readout', 0.15)]:
        print('   %-18s s=%.2f   %.2e' % (kd, s, check_classical_map(kd, s, rng)))
    print('\nO(2) acts on sigma_y by the determinant: U^T Y U = det(U) Y.  Hence for a')
    print('single-site Y the squared estimator is DETERMINISTIC (zero variance in o^2) --')
    print('the extreme case of the R proportional-to-identity fact of Prop. d2complex.')
    U = haar_O2(4000, rng)
    dev = np.abs(np.einsum('bij,jk,bkl->bil', U.transpose(0, 2, 1), PY, U) - np.linalg.det(np.real(U))[:, None, None] * PY[None]).max()
    print('   max |U^T Y U - det(U) Y| over 4000 draws = %.2e   (fails for X, Z)' % dev)
    n = 6
    p = 0.9
    _, psi = ground_state(tfim(n, 1.0, 1.0))
    Ks1 = depolarizing(2, p)
    SH = 400000
    print(f'\n--- Part II: real basis, O(2)^ox{n}, depolarizing p={p}, TFIM ground state ---')
    print('   Pauli          wt    E[o] sampled       true      resid    E[o^2] sampled      (2f_1^2)^-k    resid')
    for pl in ('ZIIIII', 'ZZIIII', 'XIZIII', 'ZZZZII', 'XXZZII'):
        smp, info = run_local(psi, n, pl, Ks1, 'depolarizing', p, SH, rng)
        m, mci, _v, _vci = ci(smp)
        true = float(np.real(psi.conj() @ (pauli_matrix(pl) @ psi)))
        m2, m2ci = moment2(smp)
        pred = predict_second_moment(Ks1, pl)
        wt = sum((1 for c in pl if c != 'I'))
        print('   %-14s %d   %+8.5f+-%.5f  %+8.5f  %s   %9.4f+-%7.4f %12.4f  %s' % (pl, wt, m, mci, true, _resid(m, true, mci), m2, m2ci, pred, _resid(m2, pred, m2ci)))
    print('\n--- (3/2)^k: local orthogonal vs local unitary, MEASURED ---')
    print('   (heavier tails at large wt, so shots are scaled with wt)')
    print('   wt    E[o^2] orthogonal      E[o^2] unitary        ratio            (3/2)^k   dev')
    for wt in (1, 2, 3, 4):
        pl = 'Z' * wt + 'I' * (n - wt)
        sh = SH * (1 if wt <= 2 else 4)
        so, _ = run_local(psi, n, pl, Ks1, 'depolarizing', p, sh, rng, ensemble='O')
        su, _ = run_local(psi, n, pl, Ks1, 'depolarizing', p, sh, rng, ensemble='U')
        mo, eo = moment2(so)
        mu, eu = moment2(su)
        r = mu / mo
        rci = r * np.sqrt((eu / mu) ** 2 + (eo / mo) ** 2)
        print('   %d   %9.4f+-%7.4f  %10.4f+-%8.4f   %7.4f+-%.4f   %7.4f   %+5.2f CI' % (wt, mo, eo, mu, eu, r, rci, 1.5 ** wt, (r - 1.5 ** wt) / rci))
    print('\n--- Part IV: complex per-qubit basis makes Y estimable ---')
    from scipy.stats import unitary_group
    Wc = unitary_group.rvs(2, random_state=np.random.default_rng(21))
    cf, inf4 = inverse_coeffs(Ks1, Wc, 'O')
    print('   basis: beta_1=%.5f  betat_1=%.5f  f_1=%.5f  q=%.5f   (a real basis has q=1/2,' % (inf4['beta1'], inf4['betat1'], inf4['f1'], inf4['q']))
    print('   where the Y coefficient 1/(f_1(2q-1)) diverges and Y leaves the visible space)')
    _, psi5 = ground_state(tfim(n - 1, 1.0, 1.0))
    plus_i = np.array([1.0, 1j]) / np.sqrt(2)
    psiY = np.kron(plus_i, psi5)
    print('   state: |+i> ox |TFIM_5>, so <Y_0> = +1 exactly')
    for pl in ('YIIIII', 'XIIIII', 'YZIIII', 'YXIIII'):
        smp, _ = run_local(psiY, n, pl, Ks1, 'depolarizing', p, SH, rng, W=Wc)
        m, mci, _v, _vci = ci(smp)
        true = float(np.real(psiY.conj() @ (pauli_matrix(pl) @ psiY)))
        m2, m2ci = moment2(smp)
        pred = predict_second_moment(Ks1, pl, W=Wc)
        print('   %-8s E[o]=%+8.5f+-%.5f (true %+8.5f, %s)   E[o^2]=%9.4f  pred %9.4f  %s' % (pl, m, mci, true, _resid(m, true, mci), m2, pred, _resid(m2, pred, m2ci)))
