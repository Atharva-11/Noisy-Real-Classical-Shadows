"""Median-of-means estimator, Fact (mom)."""
import numpy as np

def median_of_means(samples, K):
    """Median of K batch means, Fact (mom)."""
    N = samples.size // K
    return np.median(samples[:K * N].reshape(K, N).mean(axis=1))

def failure_rate_pooled(pool, true_val, eps, N, K, reps, rng):
    """Measured failure probability."""
    bad = 0
    for _ in range(reps):
        idx = rng.integers(0, pool.size, N * K)
        if abs(median_of_means(pool[idx], K) - true_val) > eps:
            bad += 1
    return (bad / reps, bad)

def smallest_N_pooled(pool, true_val, eps, K, delta, reps, Ngrid, rng):
    """Smallest batch size meeting the target."""
    trace = []
    best = None
    for N in Ngrid:
        fr, bad = failure_rate_pooled(pool, true_val, eps, N, K, reps, rng)
        trace.append((N, fr, bad))
        if fr <= delta and best is None:
            best = N
    return (best, trace)

def pool_adequacy(pool_size, N, K):
    """Whether the sample pool is large enough."""
    return pool_size / (N * K)
if __name__ == '__main__':
    from channels import depolarizing, alpha_beta
    from many_body import tfim, ground_state, traceless, var_O, var_U
    from mc_depol import run_depol
    from mc_unitary import run_unitary
    n = 3
    d = 2 ** n
    p = 0.9
    Ks = depolarizing(d, p)
    _, beta = alpha_beta(Ks, d)
    beta = float(np.real(beta))
    _, psi = ground_state(tfim(n, 1.0, 1.0))
    rho = np.outer(psi, psi)
    O0 = traceless(tfim(n, 1.0, 1.0), d)
    O0 = O0 / np.sqrt(np.real(np.trace(O0 @ O0)))
    true = float(np.real(np.trace(O0 @ rho)))
    vO = var_O(O0, rho, beta, d)
    vU = var_U(O0, rho, beta, d)
    print('=== E3: median-of-means, measured ===\n')
    print(f'  n={n}  d={d}  depolarizing p={p}  beta={beta:.4f}')
    print(f'  observable: TFIM energy, unit 2-norm;  tr(O_0 rho) = {true:+.6f}')
    print(f'  exact single-shot variances: Var_O = {vO:.4f}  Var_U = {vU:.4f}  ratio = {vU / vO:.4f}')
    M, delta, eps = (1, 0.05, 0.1)
    K = max(1, int(np.ceil(2 * np.log(2 * M / delta))))
    NO = int(np.ceil(34 * vO / eps ** 2))
    NU = int(np.ceil(34 * vU / eps ** 2))
    print(f'\n  guarantee at eps={eps}, delta={delta}, M={M}: K = ceil(2 log(2M/delta)) = {K}')
    print(f'    orthogonal  N = ceil(34 Var/eps^2) = {NO:7d}   N_tot = {NO * K:8d}')
    print(f'    unitary     N = ceil(34 Var/eps^2) = {NU:7d}   N_tot = {NU * K:8d}')
    POOL = 3000000
    print(f'\n  drawing i.i.d. pools of {POOL:,} shots per protocol ...')
    poolO = run_depol(rho, O0, d, p, beta, POOL, 1, np.random.default_rng(21)).ravel()
    poolU = run_unitary(rho, O0, Ks, d, beta, POOL, np.random.default_rng(22), depol_p=p)
    print(f'    orthogonal pool: mean {poolO.mean():+.5f} (true {true:+.5f})  var {poolO.var(ddof=1):.4f} (exact {vO:.4f})')
    print(f'    unitary    pool: mean {poolU.mean():+.5f} (true {true:+.5f})  var {poolU.var(ddof=1):.4f} (exact {vU:.4f})')
    print(f'    pool/(N*K) adequacy: orthogonal {pool_adequacy(POOL, NO, K):.0f}x, unitary {pool_adequacy(POOL, NU, K):.0f}x')
    REPS = 2000
    rng = np.random.default_rng(99)
    print(f'\n  (1) does the guarantee hold?  empirical failure rate, {REPS} repetitions')
    for lbl, pool, N in (('orthogonal', poolO, NO), ('unitary', poolU, NU)):
        fr, bad = failure_rate_pooled(pool, true, eps, N, K, REPS, rng)
        se = 1.96 * np.sqrt(max(fr * (1 - fr), 1e-12) / REPS)
        print(f'     {lbl:11s} N={N:6d} K={K}: failures {bad:4d}/{REPS} = {fr:.4f}+-{se:.4f}   (delta={delta})  {('HOLDS' if fr <= delta else 'VIOLATED')}')
    print(f'\n  (2) how loose is it?  smallest N meeting (eps={eps}, delta={delta})')
    res = {}
    for lbl, pool, N in (('orthogonal', poolO, NO), ('unitary', poolU, NU)):
        grid = sorted({max(2, int(N * x)) for x in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0)})
        Nmin, trace = smallest_N_pooled(pool, true, eps, K, delta, REPS, grid, rng)
        res[lbl] = Nmin
        for Ng, fr, bad in trace:
            flag = ' <-- smallest meeting delta' if Ng == Nmin else ''
            print(f'     {lbl:11s} N={Ng:6d}  failure {bad:4d}/{REPS} = {fr:.4f}{flag}')
        if Nmin:
            print(f'     {lbl:11s} bound N={N} is looser by {N / Nmin:.1f}x')
    if res['orthogonal'] and res['unitary']:
        print(f'\n  (3) honest shot ratio at fixed (eps, delta): N_U/N_O = {res['unitary'] / res['orthogonal']:.3f}')
        print(f'      exact variance ratio Var_U/Var_O    = {vU / vO:.3f}')
        print(f'      ratio of the loose bound prefactors = {204 / 170:.3f}   <-- what Sec. comparison warns against reading as the advantage')
