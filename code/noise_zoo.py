"""The five noise models, and their depolarizing parameters."""
import numpy as np
import scipy.linalg as sla
from channels import depolarizing, amplitude_damping_1q, tensor_channel, alpha_beta, apply_channel, dag
from noise_examples import coherent_channel, readout_channel, beta_coherent, beta_readout
Z1 = np.diag([1.0, -1.0])

def dephasing_1q(q):
    """Single-qubit dephasing."""
    return [np.sqrt(1 - q / 2) * np.eye(2, dtype=complex), np.sqrt(q / 2) * Z1.astype(complex)]

def readout_bitflip_1q(q):
    """Single-qubit readout bit flip."""
    R = np.array([[1 - q, q], [q, 1 - q]], dtype=float)
    return readout_channel(R)

def coherent_family(d, eps, H=None, seed=5):
    """Coherent over-rotations at a range of angles."""
    if H is None:
        H = _fixed_hermitian(d, seed)
    return (coherent_channel(sla.expm(1j * eps * H)), H)

def _fixed_hermitian(d, seed=5):
    r = np.random.default_rng(seed)
    A = r.standard_normal((d, d)) + 1j * r.standard_normal((d, d))
    H = (A + A.conj().T) / 2
    return H / np.linalg.norm(H)

def family(name, n, s):
    """The five noise models of the paper."""
    d = 2 ** n
    if name == 'depolarizing':
        return (depolarizing(d, s), s * d + 1 - s, 'depolarizing $p$')
    if name == 'dephasing':
        return (tensor_channel([dephasing_1q(s)] * n), float(d), 'dephasing $q$')
    if name == 'amplitude damping':
        return (tensor_channel([amplitude_damping_1q(s)] * n), (1 + s) ** n, 'amp. damping $p$')
    if name == 'coherent':
        Ks, _H = coherent_family(d, s)
        return (Ks, beta_coherent(Ks[0]), 'coherent $\\varepsilon$')
    if name == 'readout':
        Ks = tensor_channel([readout_bitflip_1q(s)] * n)
        return (Ks, d * (1 - s) ** n, 'readout $q$')
    raise ValueError(name)
FAMILIES = ('depolarizing', 'dephasing', 'amplitude damping', 'coherent', 'readout')

def f_of(beta, d):
    """f(E) = 2(beta-1)/((d-1)(d+2)), Eq. (global_depol)."""
    return 2 * (beta - 1) / ((d - 1) * (d + 2))

def beta_check(n=3, strengths=(0.1, 0.35, 0.6, 0.85)):
    """beta from the definition."""
    d = 2 ** n
    out = []
    for nm in FAMILIES:
        for s in strengths:
            Ks, bpred, _ = family(nm, n, s)
            a, b = alpha_beta(Ks, d)
            out.append((nm, s, float(np.real(b)), bpred, abs(np.real(b) - bpred) / max(abs(bpred), 1e-15), abs(np.real(a) - d)))
    return out

def sampled_variance(nm, n, s, shots=400000, seed=0):
    """Variance by sampling."""
    from mc_fast import run_trajectories
    from many_body import tfim, ground_state, traceless, var_O
    d = 2 ** n
    Ks, bpred, _ = family(nm, n, s)
    _, b = alpha_beta(Ks, d)
    b = float(np.real(b))
    _, psi = ground_state(tfim(n, 1.0, 1.0))
    rho = np.outer(psi, psi)
    O0 = traceless(tfim(n, 1.0, 1.0), d)
    O0 = O0 / np.sqrt(np.real(np.trace(O0 @ O0)))
    e = run_trajectories(rho, O0, Ks, d, b, shots, 1, np.random.default_rng(seed)).ravel()
    v = e.var(ddof=1)
    mu4 = ((e - e.mean()) ** 4).mean()
    vci = 1.96 * np.sqrt(max(mu4 - v * v, 0.0) / e.size)
    m = e.mean()
    mci = 1.96 * e.std(ddof=1) / np.sqrt(e.size)
    return (v, vci, var_O(O0, rho, b, d), m, mci, float(np.real(np.trace(O0 @ rho))), b)
if __name__ == '__main__':
    print('=== E6: all five noise families ===\n')
    print('closed-form beta vs Kraus-computed beta (n=3, d=8); alpha must equal d')
    print('  family              s      beta(Kraus)  beta(closed)   rel.err   |alpha-d|')
    worst = 0.0
    for nm, s, b, bp, rel, da in beta_check():
        worst = max(worst, rel)
        print('  %-18s %.2f   %10.6f  %10.6f   %8.1e   %8.1e' % (nm, s, b, bp, rel, da))
    print('  worst relative beta deviation: %.2e' % worst)
    print('\ndephasing null test: beta must be exactly d and f exactly f(id), every q')
    d = 8
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        Ks, bp, _ = family('dephasing', 3, q)
        _, b = alpha_beta(Ks, d)
        print('   q=%.2f   beta=%.12f   f=%.12f   f(id)=%.12f' % (q, np.real(b), f_of(np.real(b), d), f_of(d, d)))
    print('\nend-to-end sampled variance vs closed form (n=3, TFIM energy, 4e5 shots)')
    print('  family              s     beta     measured Var     predicted     ratio    mean resid/CI')
    for nm in FAMILIES:
        for si, s in enumerate((0.3, 0.7)):
            v, vci, vp, m, mci, tr, b = sampled_variance(nm, 3, s, seed=500 + 7 * FAMILIES.index(nm) + si)
            print('  %-18s %.2f  %7.4f  %8.4f+-%.4f  %10.4f   %6.4f    %+6.2f' % (nm, s, b, v, vci, vp, v / vp, (m - tr) / mci))
