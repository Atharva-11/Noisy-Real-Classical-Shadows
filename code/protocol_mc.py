"""End-to-end orthogonal shadow protocol."""
import numpy as np
from scipy.stats import ortho_group
from channels import depolarizing, amplitude_damping_1q, tensor_channel, apply_channel, alpha_beta, dag

def proj_sym(A):
    """Symmetric part of an operator."""
    return 0.5 * (A + A.T)

def Minv_sym(Y, f, d):
    """The inverse shadow channel on the symmetric part."""
    return 1.0 / f * Y + (1 - 1.0 / f) * np.trace(Y) * np.eye(d) / d

def random_state(d, rng):
    """Random state."""
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = G @ dag(G)
    return rho / np.trace(rho)

def random_symmetric_obs(d, rng):
    """Random symmetric traceless observable."""
    M = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    O = M + M.T
    O = 0.5 * (O + dag(O))
    return np.real(O)

def analytic_var_O(Oobs, rho, beta, d):
    """Orthogonal variance, Eq. (varO)."""
    A = Oobs
    Asym = proj_sym(A)
    Asym0 = Asym - np.trace(A) * np.eye(d) / d
    t1 = np.real(np.trace(Asym0 @ Asym0))
    t2 = np.real(np.trace(rho @ Asym0 @ Asym0))
    t3 = np.real(np.trace(Asym0 @ rho))
    pref = (d - 1) * (d + 2) / ((d + 4) * (beta - 1))
    return pref * ((d * (d + 3) - 4 * beta) / (2 * d * (beta - 1)) * t1 + 2 * t2) - t3 ** 2

def run_protocol(rho, Oobs, Ks, d, beta, nshots, rng):
    """Run the protocol and return shadow estimates."""
    f = 2 * (beta - 1) / ((d - 1) * (d + 2))
    ests = np.empty(nshots)
    probs_buf = np.empty(d)
    for s in range(nshots):
        O = ortho_group.rvs(d, random_state=rng)
        sig = apply_channel(Ks, O @ rho @ O.T)
        p = np.real(np.diag(sig))
        p = np.clip(p, 0, None)
        p /= p.sum()
        b = rng.choice(d, p=p)
        Pb = np.zeros((d, d))
        Pb[b, b] = 1.0
        snapshot = O.T @ Pb @ O
        rho_hat = Minv_sym(snapshot, f, d)
        ests[s] = np.real(np.trace(Oobs @ rho_hat))
    return ests
if __name__ == '__main__':
    rng = np.random.default_rng(7)
    print('=== End-to-end noisy REAL classical shadows: unbiasedness + variance ===\n')
    configs = [('depol p=0.85, d=4', depolarizing(4, 0.85), 4, 400000), ('ampdamp p=0.8, d=4', tensor_channel([amplitude_damping_1q(0.8)] * 2), 4, 400000)]
    for label, Ks, d, nshots in configs:
        a, beta = alpha_beta(Ks, d)
        rho = random_state(d, rng)
        Oobs = random_symmetric_obs(d, rng)
        true_val = np.real(np.trace(Oobs @ rho))
        ests = run_protocol(rho, Oobs, Ks, d, beta, nshots, rng)
        emp_mean, emp_var = (ests.mean(), ests.var())
        an_var = analytic_var_O(Oobs, rho, beta, d)
        se = np.sqrt(emp_var / nshots)
        print(f'[{label}]  beta={beta:.3f}')
        print(f'  E[o]:  true={true_val:+.4f}  MC={emp_mean:+.4f}  (SE {se:.4f})  bias/SE = {(emp_mean - true_val) / se:+.2f}')
        print(f'  Var :  analytic={an_var:.4f}  MC={emp_var:.4f}  ratio={emp_var / an_var:.4f}\n')
