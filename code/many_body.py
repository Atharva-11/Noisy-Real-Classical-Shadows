"""Hamiltonians, ground states, and the closed-form variances."""
import numpy as np
from functools import reduce
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], float)
Z = np.diag([1.0, -1.0])
Y = np.array([[0, -1j], [1j, 0]])

def kron(*ops):
    """Kronecker product of a list of operators."""
    return reduce(np.kron, ops)

def op_on(n, sites, ops):
    """Place a single-qubit operator on one site."""
    mats = []
    for q in range(n):
        if q in sites:
            mats.append(ops[sites.index(q)])
        else:
            mats.append(I2)
    return kron(*mats)

def tfim(n, J=1.0, h=1.0, pbc=False):
    """Transverse-field Ising Hamiltonian."""
    d = 2 ** n
    H = np.zeros((d, d))
    rng = range(n) if pbc else range(n - 1)
    for i in rng:
        H -= J * op_on(n, [i, (i + 1) % n], [Z, Z])
    for i in range(n):
        H -= h * op_on(n, [i], [X])
    return H

def heisenberg_j1j2(n, J1=1.0, J2=0.5, pbc=True):
    """J1-J2 Heisenberg Hamiltonian."""
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)

    def SS(i, j):
        return sum((op_on(n, [i, j], [P, P]) for P in (X, Y, Z))) / 4.0
    r1 = range(n) if pbc else range(n - 1)
    r2 = range(n) if pbc else range(n - 2)
    for i in r1:
        H += J1 * SS(i, (i + 1) % n)
    for i in r2:
        H += J2 * SS(i, (i + 2) % n)
    return H

def chirality(n, i, j, k):
    """Three-spin chirality, Eq. (chirality)."""
    d = 2 ** n
    chi = np.zeros((d, d), dtype=complex)
    P = [X, Y, Z]
    eps = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}
    for (a, b, c), s in eps.items():
        chi += s * op_on(n, [i, j, k], [P[a], P[b], P[c]]) / 8.0
    return chi

def ground_state(H):
    """Ground energy and ground state."""
    w, v = np.linalg.eigh(H)
    return (w[0], v[:, 0])

def traceless(O, d):
    """Traceless part O_0, Eq. (traceless_part)."""
    return O - np.trace(O) * np.eye(d) / d

def var_O(O0, rho, beta, d):
    """Orthogonal variance, Eq. (varO)."""
    t = np.real(np.trace(O0 @ O0))
    r = np.real(np.trace(rho @ O0 @ O0))
    m = np.real(np.trace(O0 @ rho))
    return (d - 1) * (d + 2) / ((d + 4) * (beta - 1)) * ((d * (d + 3) - 4 * beta) / (2 * d * (beta - 1)) * t + 2 * r) - m ** 2

def var_U(O0, rho, beta, d):
    """Unitary variance, Eq. (varU)."""
    t = np.real(np.trace(O0 @ O0))
    r = np.real(np.trace(rho @ O0 @ O0))
    m = np.real(np.trace(O0 @ rho))
    return (d * d - 1) / ((d + 2) * (beta - 1)) * ((d + d * d - 2 * beta) / (d * (beta - 1)) * t + 2 * r) - m ** 2

def kappa(O0, rho, beta, d):
    """kappa, Eq. (kappa_def)."""
    t = np.real(np.trace(O0 @ O0))
    r = np.real(np.trace(rho @ O0 @ O0))
    return d * t / (4 * (beta - 1) * r)

def ratio_predicted(k):
    """Second-moment ratio, Eq. (ratio_master)."""
    return (2 * k + 1) / (k + 1)

def norm_profile(O0):
    """Q = d sum c^2 / ||O_0||_inf^2, Eq. (Q_extensive)."""
    t = np.real(np.trace(O0 @ O0))
    s = np.max(np.abs(np.linalg.eigvalsh(O0))) ** 2
    return t / s

def beta_depol(d, p):
    """beta for depolarizing noise."""
    return p * d + 1 - p

def rho_L(d, beta):
    """The large-kappa reference state."""
    return 2 * (d + 1) * (d + 4) * (d * d + d - 2 * beta) / ((d + 2) ** 2 * (d * d + 3 * d - 4 * beta))

def rho_S(d, beta):
    """The small-kappa reference state."""
    return (d + 1) * (d + 4) / (d + 2) ** 2

def kappa_exact(O0, rho, beta, d):
    """kappa evaluated exactly."""
    t = np.real(np.trace(O0 @ O0))
    r = np.real(np.trace(rho @ O0 @ O0))
    return (d * d + 3 * d - 4 * beta) * t / (4 * d * (beta - 1) * r)

def ratio_exact_identity(O0, rho, beta, d):
    """The ratio identity, checked exactly."""
    k = kappa_exact(O0, rho, beta, d)
    return (rho_L(d, beta) * k + rho_S(d, beta)) / (k + 1)

def second_moments(O0, rho, beta, d):
    """Orthogonal and unitary second moments."""
    m = np.real(np.trace(O0 @ rho))
    return (var_O(O0, rho, beta, d) + m * m, var_U(O0, rho, beta, d) + m * m)
