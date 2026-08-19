"""The noisy orthogonal shadow channel, Eq. (global_general)."""
import numpy as np
from twirl_engine import twirl
from channels import depolarizing, amplitude_damping_1q, tensor_channel, apply_channel, adjoint_channel, alpha_beta, dag

def ptrace1(M, d):
    """Partial trace over the first factor."""
    M4 = M.reshape(d, d, d, d)
    return np.einsum('aoai->oi', M4)

def proj_sym(A):
    """Symmetric part of an operator."""
    return 0.5 * (A + A.T)

def noisy_global_shadow_channel(A, Ks, d):
    """The shadow channel, built directly from the twirl."""
    Estar = adjoint_channel(Ks)
    out = np.zeros((d, d), dtype=complex)
    AtimesI = np.kron(A, np.eye(d))
    for b in range(d):
        Pb = np.zeros((d, d))
        Pb[b, b] = 1.0
        EPb = apply_channel(Estar, Pb)
        X = np.kron(EPb, Pb)
        T2, _, _ = twirl(X, d, 2)
        out += ptrace1(AtimesI @ T2, d)
    return out

def analytic_depol_form(A, beta, d):
    """Its closed form, Eq. (global_general)."""
    f = 2 * (beta - 1) / ((d - 1) * (d + 2))
    return (f * proj_sym(A) + (1 - f) * np.trace(A) * np.eye(d) / d, f)
if __name__ == '__main__':
    rng = np.random.default_rng(3)
    print('=== Noisy global orthogonal shadow channel  M(A) =?= f*A_sym + (1-f)Tr(A)I/d ===')
    for label, Ks, d in [('identity  d=4', tensor_channel([[np.eye(2)]] * 2), 4), ('depol p=.8 d=4', depolarizing(4, 0.8), 4), ('depol p=.6 d=8', depolarizing(8, 0.6), 8), ('ampdamp p=.7 d=4', tensor_channel([amplitude_damping_1q(0.7)] * 2), 4)]:
        a, beta = alpha_beta(Ks, d)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        M = noisy_global_shadow_channel(A, Ks, d)
        Aan, f = analytic_depol_form(A, beta, d)
        err = np.linalg.norm(M - Aan) / np.linalg.norm(Aan)
        print(f'{label}: beta={beta:.4f}  f={f:.6f}  rel.err={err:.2e}')
    for d in [4, 8, 16]:
        f_id = 2 * (d - 1) / ((d - 1) * (d + 2))
        print(f'noiseless d={d}: f(I)=2(beta-1)/((d-1)(d+2)) with beta=d  ->  {f_id:.6f}   [2/(d+2)={2 / (d + 2):.6f}]   [West strength p=d/(d+2)={d / (d + 2):.6f}]')
