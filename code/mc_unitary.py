"""Monte-Carlo unitary-ensemble shadows, for the comparison."""
import numpy as np
from channels import apply_channel, alpha_beta

def haar_unitary(batch, d, rng):
    """Haar-random unitary."""
    A = rng.standard_normal((batch, d, d)) + 1j * rng.standard_normal((batch, d, d))
    Q, R = np.linalg.qr(A)
    ph = np.diagonal(R, axis1=1, axis2=2)
    return Q * (ph / np.abs(ph))[:, None, :]

def f_unitary(beta, d):
    """The unitary depolarizing parameter f_U."""
    return (beta - 1) / ((d - 1) * (d + 1))

def _channel_diag_batch(Ks, Xb):
    out = np.zeros((Xb.shape[0], Xb.shape[1]), dtype=complex)
    for K in Ks:
        Y = np.einsum('xy,iyz,wz->ixw', K, Xb, np.conj(K), optimize=True)
        out += np.diagonal(Y, axis1=1, axis2=2)
    return np.real(out)

def run_unitary(rho, O0, Ks, d, beta, N, rng, batch=None, depol_p=None):
    """Run the unitary-ensemble protocol."""
    if batch is None:
        batch = max(16, min(2000, int(8400000.0 / (d * d))))
    fU = f_unitary(beta, d)
    Ohat = O0 / fU
    out = np.empty(N)
    done = 0
    while done < N:
        B = min(batch, N - done)
        U = haar_unitary(B, d, rng)
        Ud = np.conj(U.transpose(0, 2, 1))
        if depol_p is not None:
            pr = np.real(np.einsum('bij,jk,bik->bi', U.conj(), rho, U, optimize=True))
            pr = depol_p * pr + (1.0 - depol_p) / d
        else:
            X = np.einsum('ixy,yz,iwz->ixw', U, rho, np.conj(U), optimize=True)
            pr = _channel_diag_batch(Ks, X)
        np.clip(pr, 0, None, out=pr)
        pr /= pr.sum(1, keepdims=True)
        k = (rng.random(B)[:, None] < np.cumsum(pr, 1)).argmax(1)
        v = np.take_along_axis(Ud, k[:, None, None], axis=2)[:, :, 0]
        out[done:done + B] = np.real(np.einsum('bi,ij,bj->b', v.conj(), Ohat, v, optimize=True))
        done += B
    return out

def var_ci(samples, z=1.96):
    """Variance and 95% confidence interval."""
    n = samples.size
    v = samples.var(ddof=1)
    mu4 = ((samples - samples.mean()) ** 4).mean()
    return (v, z * np.sqrt(max(mu4 - v * v, 0.0) / n))

def mean_ci(samples, z=1.96):
    """Mean and 95% confidence interval."""
    return (samples.mean(), z * samples.std(ddof=1) / np.sqrt(samples.size))
if __name__ == '__main__':
    from channels import depolarizing, amplitude_damping_1q, tensor_channel
    from many_body import var_O as mvO, var_U as mvU, traceless
    from mc_fast import run_trajectories
    from mc_depol import run_depol
    rng = np.random.default_rng(3)
    print('=== E4: global unitary protocol sampled, head-to-head with the orthogonal one ===\n')
    print('both ensembles run on the SAME (rho, O); Var_O from mc_fast, Var_U from here')
    print('   n   d   noise            Var_O meas      Var_O exact   Var_U meas       Var_U exact   ratio meas       exact')
    SH = 300000
    for n, kind, s in [(2, 'depol', 0.9), (3, 'depol', 0.9), (4, 'depol', 0.9), (2, 'ampdamp', 0.75), (3, 'depol', 0.6)]:
        d = 2 ** n
        Ks = depolarizing(d, s) if kind == 'depol' else tensor_channel([amplitude_damping_1q(s)] * n)
        _, beta = alpha_beta(Ks, d)
        beta = float(np.real(beta))
        r = np.random.default_rng(100 + n)
        G = r.standard_normal((d, d)) + 1j * r.standard_normal((d, d))
        rho = G @ np.conj(G.T)
        rho /= np.trace(rho)
        M = r.standard_normal((d, d))
        O = (M + M.T) / 2
        O0 = traceless(O, d)
        if kind == 'depol':
            so = run_depol(rho, O0, d, s, beta, SH, 1, rng).ravel()
        else:
            so = run_trajectories(rho, O0, Ks, d, beta, SH, 1, rng).ravel()
        su = run_unitary(rho, O0, Ks, d, beta, SH, rng, depol_p=s if kind == 'depol' else None)
        vo, eo = var_ci(so)
        vu, eu = var_ci(su)
        exO = mvO(O0, rho, beta, d)
        exU = mvU(O0, rho, beta, d)
        rm = vu / vo
        rci = rm * np.sqrt((eu / vu) ** 2 + (eo / vo) ** 2)
        print('   %d %3d   %-8s %.2f  %9.4f+-%6.4f  %11.4f  %9.4f+-%6.4f  %11.4f  %6.4f+-%.4f   %6.4f' % (n, d, kind, s, vo, eo, exO, vu, eu, exU, rm, rci, exU / exO))
        mo, moci = mean_ci(so)
        mu, muci = mean_ci(su)
        tr = float(np.real(np.trace(O0 @ rho)))
        print('          means: orthogonal %+.5f+-%.5f  unitary %+.5f+-%.5f  true %+.5f' % (mo, moci, mu, muci, tr))
