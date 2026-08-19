"""Random (state, observable) families for the ratio plots."""
import numpy as np
from many_body import tfim, ground_state, traceless, norm_profile, beta_depol, kappa_exact, ratio_exact_identity, second_moments, rho_L, rho_S, var_O, var_U, op_on, X as PX, Z as PZ, I2

def _unit2(O0):
    nrm = np.sqrt(np.real(np.trace(O0 @ O0)))
    return O0 / nrm if nrm > 0 else O0

def obs_gue_sym(d, rng):
    """Symmetric GUE observable."""
    M = rng.standard_normal((d, d))
    return _unit2(traceless((M + M.T) / 2, d))

def obs_west(d, rng):
    """The observable family of West et al."""
    M = rng.uniform(-1.0, 1.0, size=(d, d))
    return _unit2(traceless((M + M.T) / 2, d))

def obs_rank_r(d, rng, r=1):
    """Rank-r projector observable."""
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    P = Q[:, :r] @ Q[:, :r].T
    return _unit2(traceless(P, d))

def obs_pauli_k(d, rng, k=2, terms=None):
    """Weight-k Pauli observable."""
    n = int(round(np.log2(d)))
    if terms is None:
        terms = 2 * n
    O = np.zeros((d, d))
    for _ in range(terms):
        w = rng.integers(1, k + 1)
        sites = sorted(rng.choice(n, size=min(w, n), replace=False).tolist())
        ops = [PX if rng.random() < 0.5 else PZ for _ in sites]
        O = O + rng.standard_normal() * op_on(n, sites, ops)
    return _unit2(traceless(O, d))

def obs_tfim(d, rng=None):
    """TFIM Hamiltonian as an observable."""
    n = int(round(np.log2(d)))
    return _unit2(traceless(tfim(n, 1.0, 1.0), d))

def state_haar_mixed(d, rng):
    """Haar-random mixed state."""
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    r = G @ np.conj(G.T)
    return r / np.trace(r)

def state_haar_pure(d, rng):
    """Haar-random pure state."""
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    return np.outer(v, v)

def state_tfim_ground(d, rng=None):
    """TFIM ground state."""
    n = int(round(np.log2(d)))
    _, psi = ground_state(tfim(n, 1.0, 1.0))
    return np.outer(psi, psi)

def ratio_ensemble(n, p=0.9, n_inst=500, obs='gue_sym', state='haar_mixed', seed=0):
    """Second-moment ratio over an instance family."""
    d = 2 ** n
    beta = beta_depol(d, p)
    fO = {'gue_sym': obs_gue_sym, 'west': obs_west, 'tfim': obs_tfim, 'rank1': lambda dd, r: obs_rank_r(dd, r, 1)}[obs]
    fS = {'haar_mixed': state_haar_mixed, 'haar_pure': state_haar_pure, 'tfim': state_tfim_ground}[state]
    rng = np.random.default_rng((seed, n))
    out = np.empty(n_inst)
    for i in range(n_inst):
        O0 = fO(d, rng)
        rho = fS(d, rng)
        out[i] = var_U(O0, rho, beta, d) / var_O(O0, rho, beta, d)
    return out

def ghz(n):
    """GHZ state."""
    d = 2 ** n
    v = np.zeros(d)
    v[0] = v[-1] = 1 / np.sqrt(2)
    return v

def _aligned_rank_r(d, rng, r):
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    V = Q[:, :r]
    O0 = _unit2(traceless(V @ V.T, d))
    c = rng.standard_normal(r)
    c /= np.linalg.norm(c)
    v = V @ c
    return (O0, np.outer(v, v))

def kappa_scatter(ns=(2, 3, 4, 5, 6), p=0.9, per_family=40, seed=1):
    """kappa against the ratio, Eq. (kappa_def)."""
    rows = []
    for n in ns:
        d = 2 ** n
        beta = beta_depol(d, p)
        rng = np.random.default_rng((seed, n))

        def add(O0, rho, fam):
            if np.real(np.trace(rho @ O0 @ O0)) <= 1e-12:
                return
            MO, MU = second_moments(O0, rho, beta, d)
            rows.append((kappa_exact(O0, rho, beta, d), MU / MO, d, fam, norm_profile(O0)))
        for r in sorted({1, 2, max(1, d // 8), max(1, d // 4)}):
            for _ in range(max(1, per_family // 4)):
                O0, rho = _aligned_rank_r(d, rng, r)
                add(O0, rho, 'fidelity / aligned projector')
        g = ghz(n)
        add(_unit2(traceless(np.outer(g, g), d)), np.outer(g, g), 'GHZ fidelity')
        for _ in range(per_family):
            add(obs_gue_sym(d, rng), state_haar_pure(d, rng), 'dense symmetric')
        for _ in range(per_family):
            add(obs_pauli_k(d, rng, 2), state_haar_pure(d, rng), 'local Pauli sum')
        for r in sorted({1, max(1, d // 4)}):
            for _ in range(max(1, per_family // 2)):
                add(obs_rank_r(d, rng, r), state_haar_pure(d, rng), 'unaligned projector')
        add(obs_tfim(d), state_tfim_ground(d), 'TFIM energy')
    return rows

def collapse_residual(rows, p=0.9):
    """Residual of the collapse onto Eq. (ratio_master)."""
    worst = 0.0
    for k, R, d, _f, _q in rows:
        beta = beta_depol(d, p)
        pred = (rho_L(d, beta) * k + rho_S(d, beta)) / (k + 1)
        worst = max(worst, abs(R - pred) / abs(pred))
    return worst
if __name__ == '__main__':
    print('=== E5: instance ensembles ===\n')
    print('Var_U/Var_O over 500 instances per n, dense symmetric obs, Haar mixed states, p=0.9')
    print('   n     d      median      IQR              min      max')
    for n in range(2, 8):
        r = ratio_ensemble(n, n_inst=500, seed=0)
        q1, q2, q3 = np.percentile(r, [25, 50, 75])
        print('  %2d %5d   %8.5f   [%7.5f, %7.5f]   %7.5f  %7.5f' % (n, 2 ** n, q2, q1, q3, r.min(), r.max()))
    print("\nWest et al.'s Uniform[-1,1] observable distribution, same protocol:")
    for n in (2, 4, 6):
        r = ratio_ensemble(n, n_inst=500, obs='west', seed=0)
        print('   n=%d  median %.5f   IQR [%.5f, %.5f]' % (n, np.median(r), *np.percentile(r, [25, 75])))
    rows = kappa_scatter()
    ks = np.array([r[0] for r in rows])
    print('\nkappa scatter: %d instances, kappa spans %.3g .. %.3g (%.1f decades)' % (len(rows), ks.min(), ks.max(), np.log10(ks.max() / ks.min())))
    print('collapse onto the exact identity: max relative deviation = %.2e' % collapse_residual(rows))
    for f in sorted({r[3] for r in rows}):
        kk = np.array([r[0] for r in rows if r[3] == f])
        RR = np.array([r[1] for r in rows if r[3] == f])
        print('   %-22s n=%3d   kappa %.3g .. %.3g   ratio %.3f .. %.3f' % (f, len(kk), kk.min(), kk.max(), RR.min(), RR.max()))
