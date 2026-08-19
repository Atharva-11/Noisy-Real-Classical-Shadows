"""Complex-basis second moment, Eq. (complex_R)."""
import numpy as np, itertools
from weingarten_core import perm_operator, omega_operator, PERM_ORDER, S3, OMEGA_ORDER
from channels import adjoint_channel, apply_channel

def diagrams(d):
    """The fifteen Brauer diagrams at order three."""
    ops = [perm_operator(S3[p], d, 3) for p in PERM_ORDER]
    ops += [omega_operator(o, i, d, 3) for o, i in OMEGA_ORDER]
    return ops

def measurement_invariants(Ks, W, d):
    """The six invariants of Eq. (complex_invariants)."""
    Estar = adjoint_channel(Ks)
    inv = dict(d=float(d), beta=0j, betat=0j, gamma=0j, delta=0j)
    for k in range(d):
        w = W[:, k]
        ws = w.conj()
        Pw = np.outer(w, w.conj())
        A = apply_channel(Estar, Pw)
        ov = np.sum(w.conj() * ws)
        inv['beta'] += np.vdot(w, A @ w)
        inv['betat'] += np.vdot(ws, A @ ws)
        inv['gamma'] += np.trace(A) * abs(ov) ** 2
        inv['delta'] += ov * np.vdot(ws, A @ w)
    return inv

def coefficients(d, beta, betat, gamma, delta):
    """Coefficients of Eq. (complex_R), given by Eq. (complex_coeffs)."""
    D = d * (d - 2) * (d - 1) * (d + 2) * (d + 4)
    dl = delta
    dlc = np.conj(delta)
    A = 2 * ((d ** 2 + d - 4) * beta - (d - 4) * betat - d ** 2 - 2 * d * np.real(dl) + 2 * gamma) / D
    E = 2 * ((d ** 2 + d - 4) * betat - (d - 4) * beta - d ** 2 - 2 * d * np.real(dl) + 2 * gamma) / D
    B = 2 * (-2 * d * (beta + betat) + (d ** 2 + 2 * d - 4) * dl + 4 * dlc - (d + 2) * gamma + 4 * d) / D
    C = np.conj(B)
    s = (d ** 3 + 2 * d ** 2 - 4 * d - 2 * d * (beta + betat) - (d + 2) * gamma + 4 * (dl + dlc)) / D
    st = ((d ** 2 + 3 * d - 2) * gamma - 2 * d * (d + 2) + 8 * (beta + betat) - 2 * (d + 2) * (dl + dlc)) / D
    return (A, B, C, E, s, st)

def R_operator(Ohat, d, inv):
    """The operator R of Eq. (complex_R)."""
    A, B, C, E, s, st = coefficients(d, inv['beta'], inv['betat'], inv['gamma'], inv['delta'])
    OT = Ohat.T
    return (s * np.trace(Ohat @ Ohat) + st * np.trace(Ohat @ OT)) * np.eye(d) + A * Ohat @ Ohat + B * OT @ Ohat + C * Ohat @ OT + E * OT @ OT

def second_moment(rho, Ohat, Ks, W, d):
    """E[o^2] = tr[rho R]."""
    return np.real(np.trace(rho @ R_operator(Ohat, d, measurement_invariants(Ks, W, d))))

def invariants_depol(d, p, alpha_r):
    """The invariants for depolarizing noise, Eq. (depol_complex)."""
    return dict(beta=p * d + 1 - p, betat=p * alpha_r + 1 - p, gamma=float(alpha_r), delta=complex((p + (1 - p) / d) * alpha_r))

def Mdag_inv_depol(X, d, p, alpha_r):
    """M^{-1,dag} for depolarizing noise."""
    iv = invariants_depol(d, p, alpha_r)
    beta, betat = (iv['beta'], iv['betat'])
    f = (beta + betat - 2) / ((d - 1) * (d + 2))
    q = (beta * (d + 1) - d - betat) / (d * (beta + betat - 2))
    Y = X / f + (1 - 1 / f) * np.trace(X) * np.eye(d) / d
    if abs(2 * q - 1) < 1e-12:
        return (Y + Y.T) / 2
    return (q * Y - (1 - q) * Y.T) / (2 * q - 1)

def variance_depol(O0, rho, d, p, alpha_r):
    """Variance under depolarizing noise, in closed form."""
    Oh = Mdag_inv_depol(np.asarray(O0, dtype=complex), d, p, alpha_r)
    R = R_operator(Oh, d, invariants_depol(d, p, alpha_r))
    m2 = float(np.real(np.trace(rho @ R)))
    return m2 - float(np.real(np.trace(np.asarray(O0) @ rho))) ** 2
if __name__ == '__main__':
    from exact_tools import exact_mean_var, shadow_superop
    from channels import dag
    from scipy.stats import unitary_group
    rng = np.random.default_rng(0)
    print('closed form  E[o^2]=Tr[rho R]  vs exact twirl (random complex CPTP + complex basis):')
    mx = 0.0
    for d in (4, 5, 6, 7):
        for t in range(2):
            Ks = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(3)]
            S = sum((K.conj().T @ K for K in Ks))
            L = np.linalg.cholesky(np.linalg.inv(S))
            Ks = [K @ L for K in Ks]
            G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            rho = G @ dag(G)
            rho /= np.trace(rho)
            Ms = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
            O = (Ms + Ms.conj().T) / 2
            O0 = O - np.trace(O) * np.eye(d) / d
            W = unitary_group.rvs(d, random_state=rng)
            Oh = (np.linalg.pinv(shadow_superop(Ks, d, W)).conj().T @ O0.reshape(d * d)).reshape(d, d)
            cf = second_moment(rho, Oh, Ks, W, d)
            Eo, V = exact_mean_var(O0, rho, Ks, d, W)
            ex = np.real(V + Eo ** 2)
            mx = max(mx, abs(cf - ex) / abs(ex))
        print(f'  d={d}: rel.err={abs(cf - ex) / abs(ex):.2e}')
    print(f'max relative error over d=4..7: {mx:.2e}')
    import sympy as sp
    dd, be = sp.symbols('d beta', positive=True)
    D = dd * (dd - 2) * (dd - 1) * (dd + 2) * (dd + 4)
    A = 2 * ((dd ** 2 + dd - 4) * be - (dd - 4) * be - dd ** 2 - 2 * dd * be + 2 * dd) / D
    s = (dd ** 3 + 2 * dd ** 2 - 4 * dd - 2 * dd * (2 * be) - (dd + 2) * dd + 4 * (2 * be)) / D
    st = ((dd ** 2 + 3 * dd - 2) * dd - 2 * dd * (dd + 2) + 8 * (2 * be) - 2 * (dd + 2) * (2 * be)) / D
    ratio = sp.simplify(4 * A / (s + st))
    print('real-basis operator/scalar ratio =', sp.factor(ratio), ' (expected 4d(beta-1)/(d(d+3)-4beta))')
