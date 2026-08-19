"""Makes every figure in the paper. Run: python make_figures.py"""
import numpy as np, os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from channels import depolarizing, amplitude_damping_1q, tensor_channel, alpha_beta, dag
from protocol_mc import run_protocol, random_state, random_symmetric_obs, analytic_var_O, proj_sym
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 600, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'lines.linewidth': 1.3, 'savefig.bbox': 'tight', 'pdf.fonttype': 42, 'ps.fonttype': 42})
W2 = (6.5, 2.75)
W1 = (3.9, 2.95)
OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT, exist_ok=True)
C = {'O': '#0072B2', 'U': '#D55E00', 'th': '#000000', 'acc': '#009E73'}

def save(fig, name, tight=True):
    """Write a figure to ../figures as PDF and PNG."""
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', dpi=220, facecolor='white')
    plt.close(fig)
    print('wrote', name)

def var_O(A0, rho, beta, d):
    """Orthogonal variance, Eq. (varO)."""
    t1 = np.real(np.trace(A0 @ A0))
    t2 = np.real(np.trace(rho @ A0 @ A0))
    t3 = np.real(np.trace(A0 @ rho))
    return (d - 1) * (d + 2) / ((d + 4) * (beta - 1)) * ((d * (d + 3) - 4 * beta) / (2 * d * (beta - 1)) * t1 + 2 * t2) - t3 ** 2

def var_U(A0, rho, beta, d):
    """Unitary variance, Eq. (varU)."""
    t1 = np.real(np.trace(A0 @ A0))
    t2 = np.real(np.trace(rho @ A0 @ A0))
    t3 = np.real(np.trace(A0 @ rho))
    return (d * d - 1) / ((d + 2) * (beta - 1)) * ((d + d * d - 2 * beta) / (d * (beta - 1)) * t1 + 2 * t2) - t3 ** 2

def fig_variance_ratio():
    """Figure 1: exact variance ratio over random instances."""
    from instance_ensembles import ratio_ensemble
    ns = list(range(2, 9))
    p = 0.9
    NI = 500
    med = []
    q1 = []
    q3 = []
    lo = []
    hi = []
    wmed = []
    fratio = []
    for n in ns:
        r = ratio_ensemble(n, p=p, n_inst=NI, obs='gue_sym', state='haar_mixed', seed=0)
        a, b, c, e, g = np.percentile(r, [5, 25, 50, 75, 95])
        lo.append(a)
        q1.append(b)
        med.append(c)
        q3.append(e)
        hi.append(g)
        w = ratio_ensemble(n, p=p, n_inst=NI, obs='west', state='haar_mixed', seed=0)
        wmed.append(np.median(w))
        fratio.append(2 * (2 ** n + 1) / (2 ** n + 2))
    fig, ax = plt.subplots(figsize=W1)
    ax.axhline(2, ls='--', c=C['th'], lw=1, label='asymptote $2$')
    ax.fill_between(ns, lo, hi, color=C['O'], alpha=0.15, lw=0, label='5-95% of instances')
    ax.fill_between(ns, q1, q3, color=C['O'], alpha=0.35, lw=0, label='interquartile')
    ax.plot(ns, fratio, ':', c='0.25', lw=1.6, zorder=6, label='$f_{\\mathbb{O}}/f_{\\mathbb{U}}=2(d{+}1)/(d{+}2)$')
    ax.plot(ns, med, 'o-', c=C['O'], ms=5, label='median exact $\\mathrm{Var}_{\\mathbb{U}}/\\mathrm{Var}_{\\mathbb{O}}$')
    ax.plot(ns, wmed, '^--', c=C['acc'], ms=4, lw=1, mfc='none', label='median, $\\mathrm{Unif}[-1,1]$ observables')
    from mc_unitary import run_unitary, var_ci
    from mc_depol import run_depol
    from many_body import traceless, var_O as mvO, var_U as mvU
    SHOTS = {2: 250000, 3: 250000, 4: 250000, 5: 250000, 6: 250000, 7: 68000, 8: 68000}
    import json as _json, os as _os
    _CF = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'diamond_cache.json')
    try:
        _cache = _json.load(open(_CF))
    except Exception:
        _cache = {}
    sn = []
    sr = []
    sci = []
    for n in ns:
        SH = SHOTS[n]
        key = f'{n}:{SH}'
        if key in _cache:
            rr, cc, ex = _cache[key]
            print(f'  fig1: n={n} (cached) ratio {rr:.4f}+-{cc:.4f}   exact {ex:.4f}   resid {(rr - ex) / cc:+.2f} CI')
        else:
            d = 2 ** n
            beta = p * d + 1 - p
            r = np.random.default_rng(900 + n)
            G = r.standard_normal((d, d)) + 1j * r.standard_normal((d, d))
            rho = G @ np.conj(G.T)
            rho /= np.trace(rho)
            Ms = r.standard_normal((d, d))
            O0 = traceless((Ms + Ms.T) / 2, d)
            so = run_depol(rho, O0, d, p, beta, SH, 1, np.random.default_rng(31 + n)).ravel()
            su = run_unitary(rho, O0, None, d, beta, SH, np.random.default_rng(41 + n), depol_p=p)
            vo, eo = var_ci(so)
            vu, eu = var_ci(su)
            rr = vu / vo
            cc = rr * np.sqrt((eu / vu) ** 2 + (eo / vo) ** 2)
            ex = mvU(O0, rho, beta, d) / mvO(O0, rho, beta, d)
            _cache[key] = [float(rr), float(cc), float(ex)]
            _json.dump(_cache, open(_CF, 'w'), indent=1)
            print(f'  fig1: n={n} sampled ratio {rr:.4f}+-{cc:.4f}   exact (same instance) {ex:.4f}   resid {(rr - ex) / cc:+.2f} CI', flush=True)
        sn.append(n)
        sr.append(rr)
        sci.append(cc)
    ax.errorbar(sn, sr, yerr=sci, fmt='D', ms=4, c='0.2', mfc='white', capsize=2, lw=1, zorder=7, label='sampled, both protocols (1 instance)')
    ax.set_xlabel('number of qubits $n$')
    ax.set_ylabel('variance ratio')
    ax.set_title(f'Factor-of-two advantage, depolarizing $p={p}$ ({NI} instances/point)')
    ax.set_ylim(1.4, 2.08)
    ax.legend(loc='lower right', fontsize=7)
    save(fig, 'fig1_variance_ratio')
    print('  fig1: median ratio by n = ' + ', '.join((f'{m:.4f}' for m in med)))
    print('  fig1: IQR width by n    = ' + ', '.join((f'{b - a:.4f}' for a, b in zip(q1, q3))))
    print('  fig1: |median(gue)-median(west)| max = %.2e' % max((abs(a - b) for a, b in zip(med, wmed))))

def fig_convergence():
    """Not in the paper: Monte-Carlo convergence of the estimator."""
    from mc_fast import run_trajectories
    rng = np.random.default_rng(4)
    d = 4
    p = 0.85
    Ks = depolarizing(d, p)
    _, beta = alpha_beta(Ks, d)
    rho = random_state(d, rng)
    O = random_symmetric_obs(d, rng)
    true = np.real(np.trace(O @ rho))
    O0 = proj_sym(O) - np.trace(O) * np.eye(d) / d
    anvar = analytic_var_O(O, rho, beta, d)
    M = 40
    N = 250000
    E = run_trajectories(rho, O, Ks, d, beta, N, M, rng)
    step = np.unique(np.geomspace(50, N, 60).astype(int))
    cs = np.cumsum(E, axis=1)
    cs2 = np.cumsum(E ** 2, axis=1)
    ks = step.astype(float)
    rm = cs[:, step - 1] / ks
    rv = cs2[:, step - 1] / ks - rm ** 2
    mean_curve = rm.mean(0)
    se = rm.std(0) / np.sqrt(M)
    var_curve = rv.mean(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    a1.axhline(true, c=C['th'], ls='--', lw=1, label='$\\mathrm{tr}(O\\rho)$')
    a1.fill_between(step, mean_curve - 1.96 * se, mean_curve + 1.96 * se, color=C['O'], alpha=0.25, label='95% CI ($M{=}40$ runs)')
    a1.plot(step, mean_curve, c=C['O'], lw=1.3, label='running estimate (avg)')
    a1.set_xscale('log')
    a1.set_xlabel('number of shots $N$')
    a1.set_ylabel('$\\hat{o}$')
    a1.legend()
    a1.set_title('Unbiasedness')
    a2.axhline(anvar, c=C['th'], ls='--', lw=1, label='analytic $\\mathrm{Var}_{\\mathbb{O}}$')
    a2.plot(step, var_curve, c=C['acc'], lw=1.3, label='empirical variance (avg, $M{=}40$)')
    a2.set_xscale('log')
    a2.set_xlabel('number of shots $N$')
    a2.set_ylabel('variance')
    a2.legend()
    a2.set_title('Variance formula')
    save(fig, 'fig2_convergence')

def fig_local_advantage():
    """Figure 5: local Pauli seminorm, orthogonal against unitary."""
    from mc_local import run_local, predict_second_moment, moment2
    from many_body import tfim, ground_state
    ks = np.arange(1, 7)
    p = 0.9
    Ks1 = depolarizing(2, p)
    _, b1 = alpha_beta(Ks1, 2)
    b1 = float(np.real(b1))
    fO = (b1 - 1) / 2
    fU = (b1 - 1) / 3
    semiO = (2 * fO ** 2) ** (-ks.astype(float))
    semiU = (3 * fU ** 2) ** (-ks.astype(float))
    rng = np.random.default_rng(11)
    nref = 6
    _, psi = ground_state(tfim(nref, 1.0, 1.0))
    SH = 250000
    mk = []
    mo = []
    eo = []
    mu = []
    eu = []
    for k in (1, 2, 3, 4):
        pl = 'Z' * k + 'I' * (nref - k)
        sh = SH * (1 if k <= 2 else 3)
        so, _ = run_local(psi, nref, pl, Ks1, 'depolarizing', p, sh, rng, ensemble='O')
        su, _ = run_local(psi, nref, pl, Ks1, 'depolarizing', p, sh, rng, ensemble='U')
        a, b = moment2(so)
        c, e = moment2(su)
        mk.append(k)
        mo.append(a)
        eo.append(b)
        mu.append(c)
        eu.append(e)
        print(f'  fig3: k={k} n={nref}  orth {a:9.4f}+-{b:.4f} (pred {predict_second_moment(Ks1, pl):9.4f})  unit {c:10.4f}+-{e:.4f} (pred {predict_second_moment(Ks1, pl, ensemble='U'):10.4f})')
    nn = [4, 6, 8, 10]
    nr = []
    nrci = []
    for n in nn:
        _, ps = ground_state(tfim(n, 1.0, 1.0))
        pl = 'ZZ' + 'I' * (n - 2)
        sh = 150000 if n >= 10 else 200000
        so, _ = run_local(ps, n, pl, Ks1, 'depolarizing', p, sh, rng, ensemble='O')
        su, _ = run_local(ps, n, pl, Ks1, 'depolarizing', p, sh, rng, ensemble='U')
        a, b = moment2(so)
        c, e = moment2(su)
        r = c / a
        nr.append(r)
        nrci.append(r * np.sqrt((e / c) ** 2 + (b / a) ** 2))
        print(f'  fig3: n={n} wt=2 ratio {r:.4f}+-{nrci[-1]:.4f}  (3/2)^2=2.25')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    a1.semilogy(ks, semiU, '-', c=C['U'], lw=1.3, label='unitary $(3f_{\\mathbb{U}}^2)^{-k}$')
    a1.semilogy(ks, semiO, '-', c=C['O'], lw=1.3, label='orthogonal $(2f_{\\mathbb{O}}^2)^{-k}$')
    a1.errorbar(mk, mu, yerr=eu, fmt='s', ms=4, c=C['U'], mfc='white', capsize=2, lw=1, label='sampled, unitary')
    a1.errorbar(mk, mo, yerr=eo, fmt='o', ms=4, c=C['O'], mfc='white', capsize=2, lw=1, label='sampled, orthogonal')
    a1.set_xlabel('Pauli weight $k$')
    a1.set_ylabel('$\\mathbb{E}[\\hat o^2]=\\|P\\|^2$')
    a1.set_title(f'Local Pauli seminorm ($p={p}$, $n={nref}$)')
    a1.legend(fontsize=7)
    a2.plot(ks, 1.5 ** ks, '-', c=C['th'], lw=1.2, label='$(3/2)^k$ (exact)')
    a2.errorbar(mk, np.array(mu) / np.array(mo), yerr=np.array(mu) / np.array(mo) * np.sqrt((np.array(eu) / np.array(mu)) ** 2 + (np.array(eo) / np.array(mo)) ** 2), fmt='o', ms=4, c=C['acc'], mfc='white', capsize=2, lw=1, label=f'sampled ratio, $n={nref}$')
    a2.errorbar([2.0 + 0.1 * (i - 1.5) for i in range(len(nn))], nr, yerr=nrci, fmt='^', ms=4, c='0.45', mfc='white', capsize=2, lw=1, label='$k{=}2$ at $n=4,6,8,10$')
    a2.set_yscale('log')
    a2.set_xlabel('Pauli weight $k$')
    a2.set_ylabel('sample-complexity ratio')
    a2.set_title('Exponential advantage, measured')
    a2.legend(fontsize=7, loc='upper left')
    save(fig, 'fig3_local_advantage')

def fig_noise_dependence():
    """Figure 4: f(E) for the five noise models."""
    from noise_zoo import family, f_of, FAMILIES, sampled_variance
    n = 3
    d = 2 ** n
    ss = np.linspace(0.0, 1.0, 41)
    STY = {'depolarizing': (C['O'], '-'), 'dephasing': (C['th'], '-'), 'amplitude damping': (C['U'], '-'), 'coherent': (C['acc'], '-'), 'readout': ('0.45', '-')}
    MCS = {'depolarizing': (0.2, 0.45, 0.7, 0.9), 'dephasing': (0.2, 0.45, 0.7, 0.9), 'amplitude damping': (0.2, 0.45, 0.7, 0.9), 'coherent': (0.2, 0.45, 0.7, 0.9), 'readout': (0.1, 0.25, 0.7, 0.85)}
    fig, a1 = plt.subplots(figsize=W1)
    for nm in FAMILIES:
        col, ls = STY[nm]
        fs = []
        lab = None
        for s in ss:
            _K, b, lab = family(nm, n, s)
            fs.append(f_of(b, d))
        a1.plot(ss, fs, ls, c=col, lw=1.3, label=lab)
    a1.axhline(f_of(d, d), ls=':', c=C['th'], lw=1)
    a1.axhline(0, c='0.7', lw=0.8)
    a1.annotate('$f(\\mathrm{id})=\\frac{2}{d+2}$', xy=(0.02, f_of(d, d) * 1.06), fontsize=7)
    a1.plot([0.5], [0.0], 'o', c='0.25', ms=4, zorder=5)
    a1.annotate('$\\beta=1$ at $q=\\frac{1}{2}$: not invertible', xy=(0.5, 0.0), xytext=(0.56, -0.052), fontsize=6.4, color='0.25', arrowprops=dict(arrowstyle='->', lw=0.7, color='0.25', shrinkA=0, shrinkB=3))
    a1.set_xlabel('noise strength')
    a1.set_ylabel('depolarizing parameter $f(\\mathcal{E})$')
    a1.set_title(f'All five noise models ($n={n}$)')
    a1.legend(fontsize=6.4, loc='lower left', ncol=2, columnspacing=0.9, handlelength=1.4)
    a1.set_ylim(-0.155, 0.235)
    for nm in FAMILIES:
        for si, s in enumerate(MCS[nm]):
            v, vci, vp, m, mci, tr, b = sampled_variance(nm, n, s, shots=250000, seed=700 + 37 * FAMILIES.index(nm) + si)
            print(f'  fig4: {nm:18s} s={s:.2f} beta={b:7.4f} Var {v:9.4f} vs {vp:9.4f} ratio {v / vp:.4f}+-{vci / vp:.4f}  mean resid {(m - tr) / mci:+.2f} CI')
    save(fig, 'fig4_noise_dependence')

def fig_complex_crossover():
    """Figure 6: complex-basis crossover in the reality fraction."""
    from mc_general import run, mean_ci, var_ci
    from complex_third_moment import variance_depol, Mdag_inv_depol
    from many_body import var_U as _varU
    p = 0.8
    fig, (ax, bx) = plt.subplots(1, 2, figsize=W2)

    def paired(d, t):
        B = np.array([[np.cos(t), 1j * np.sin(t)], [1j * np.sin(t), np.cos(t)]])
        W = np.zeros((d, d), dtype=complex)
        for j in range(0, d, 2):
            W[j:j + 2, j:j + 2] = B
        return W
    NS = [(3, '#2c6fbb', 'o'), (5, '#8e44ad', 's'), (7, '#16a085', '^')]
    for n, c, _ in NS:
        d = 2 ** n
        ar = np.linspace(0, d, 200)
        f = p * (ar + d - 2) / ((d - 1) * (d + 2))
        ax.plot(ar / d, f * d, c=c, lw=1.4, label=f'$n={n}$, exact')
        ax.scatter([1], [p * (2 * d - 2) / ((d - 1) * (d + 2)) * d], c=c, zorder=5, s=25)
        ax.axhline(d * p / (d + 1), ls=(0, (4, 2)), c=c, lw=0.9, alpha=0.75)
    rngm = np.random.default_rng(77)
    SHOTS = {3: 1500000, 5: 1500000}
    for n, c, mk in NS:
        if n not in SHOTS:
            continue
        d = 2 ** n
        A = rngm.standard_normal((d, d))
        rho = A @ A.T
        rho /= np.trace(rho)
        r0 = rho - np.trace(rho) * np.eye(d) / d
        nrm = float(np.real(np.trace(r0 @ r0)))
        xs = []
        ys = []
        es = []
        for sg in (0.0, 0.25, 0.55, 1.0):
            W = paired(d, np.arccos(np.sqrt(sg)) / 2) if sg < 1 else np.eye(d, dtype=complex)
            sm = run(rho, r0.astype(complex), W, d, p, SHOTS[n], np.random.default_rng(900 + 10 * n + int(20 * sg)))
            m, ci = mean_ci(sm)
            xs.append(sg)
            ys.append(d * m / nrm)
            es.append(d * ci / nrm)
            fex = p * (sg * d + d - 2) / ((d - 1) * (d + 2))
            print(f'  fig5L: n={n} varsigma={sg:.2f}  d*f measured {d * m / nrm:.4f}+-{d * ci / nrm:.4f}   exact {d * fex:.4f}   resid {(m / nrm - fex) / max(ci / nrm, 1e-15):+.2f} CI', flush=True)
        ax.errorbar(xs, ys, yerr=es, fmt=mk, ms=4.5, c=c, mfc='white', capsize=2.5, lw=1.1, zorder=8, label=f'$n={n}$, simulated')
    from matplotlib.lines import Line2D
    ax.axhline(p, ls=':', c='0.25', lw=1.2, label='$d\\,f_{\\mathbb{U}}\\to p$ as $d\\to\\infty$')
    proxy = Line2D([], [], ls=(0, (4, 2)), c='0.45', lw=0.9, label='$d\\,f_{\\mathbb{U}}=dp/(d{+}1)$, per $n$')
    ax.set_xlabel('reality fraction $\\varsigma=\\alpha_{\\mathrm{r}}/d$')
    ax.set_ylabel('$d\\cdot f(\\mathcal{E})$')
    ax.set_title('The depolarizing parameter reaches the unitary value', fontsize=8.5)
    h, l = ax.get_legend_handles_labels()
    h.append(proxy)
    l.append(proxy.get_label())
    ax.legend(h, l, fontsize=6.4, loc='upper left', ncol=2)
    sgs = np.linspace(0.0, 1.0, 41)
    ends = {}
    for n, c, mk in NS:
        d = 2 ** n
        rg = np.random.default_rng(400 + n)
        A = rg.standard_normal((d, d))
        rho = A @ A.T
        rho /= np.trace(rho)
        Ms = rg.standard_normal((d, d))
        O0 = (Ms + Ms.T) / 2
        O0 -= np.trace(O0) * np.eye(d) / d
        O0 /= np.sqrt(np.trace(O0 @ O0))
        vU = float(np.real(_varU(O0, rho, p * d + 1 - p, d)))
        rat = [variance_depol(O0, rho, d, p, sg * d) / vU for sg in sgs]
        bx.plot(sgs, rat, c=c, lw=1.4, label=f'$n={n}$')
        ends[n] = (rat[0], rat[-1])
        print(f'  fig5R: n={n}  Var_O/Var_U  varsigma=0 {rat[0]:.4f}   varsigma=1 {rat[-1]:.4f}', flush=True)
        if n == 3:
            xs = []
            ys = []
            es = []
            for sg in (0.0, 0.5, 1.0):
                W = paired(d, np.arccos(np.sqrt(sg)) / 2) if sg < 1 else np.eye(d, dtype=complex)
                Oh = Mdag_inv_depol(O0.astype(complex), d, p, sg * d)
                sm = run(rho, Oh, W, d, p, 600000, np.random.default_rng(1300 + int(50 * sg)))
                v, ci = var_ci(sm)
                xs.append(sg)
                ys.append(v / vU)
                es.append(ci / vU)
                ex = variance_depol(O0, rho, d, p, sg * d)
                print(f'     fig5R sim: varsigma={sg:.2f}  {v / vU:.4f}+-{ci / vU:.4f}   exact {ex / vU:.4f}   resid {(v - ex) / max(ci, 1e-15):+.2f} CI', flush=True)
            bx.errorbar(xs, ys, yerr=es, fmt=mk, ms=4.5, c=c, mfc='white', capsize=2.5, lw=1.1, zorder=8, label='$n=3$, simulated')
    bx.axhline(1.0, ls=(0, (4, 2)), c='0.35', lw=1.0, label='unitary reference')
    bx.axhline(0.5, ls=':', c='0.35', lw=1.0, label='factor of two')
    bx.set_xlabel('reality fraction $\\varsigma=\\alpha_{\\mathrm{r}}/d$')
    bx.set_ylabel('$\\mathrm{Var}_{\\mathbb{O}}/\\mathrm{Var}_{\\mathbb{U}}$')
    bx.set_title('The variance does not, at fixed $d$', fontsize=8.5)
    bx.legend(fontsize=6.6, loc='upper right')
    for a in (ax, bx):
        a.set_xlim(-0.02, 1.02)
        a.annotate('complex', xy=(0.02, -0.15), xycoords='axes fraction', fontsize=6.3, color='0.4')
        a.annotate('real', xy=(0.93, -0.15), xycoords='axes fraction', fontsize=6.3, color='0.4')
    save(fig, 'fig5_complex_crossover')
    print('  fig5: Var_O/Var_U at varsigma=0 -> ' + ', '.join((f'n={n}: {ends[n][0]:.3f}' for n, _, _ in NS)) + '   (overshoots 1, falls back as d grows)')
_E = np.array([[3, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1], [2, 3, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2], [2, 1, 3, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1], [2, 1, 1, 3, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1], [1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1], [1, 2, 2, 2, 1, 3, 1, 1, 1, 1, 2, 2, 1, 1, 2], [2, 1, 2, 1, 1, 1, 3, 1, 1, 2, 2, 1, 2, 1, 2], [2, 2, 1, 1, 1, 1, 1, 3, 1, 2, 2, 2, 1, 2, 1], [2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 2, 2, 2, 2], [1, 1, 1, 2, 2, 1, 2, 2, 1, 3, 1, 2, 1, 1, 2], [1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 3, 1, 2, 2, 1], [1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 3, 2, 1, 1], [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 1, 1], [1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 3, 2], [1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 3]], float)

def fig_noise_blind_monotone():
    """Figure 3: noise-blind bias, and variance monotonicity."""
    from mc_general import run, mean_ci, var_ci
    from exact_tools import shadow_superop
    from channels import depolarizing, alpha_beta
    from many_body import traceless, var_O as _mvO
    n = 3
    d = 2 ** n
    W = np.eye(d, dtype=complex)
    rng = np.random.default_rng(2024)
    A = rng.standard_normal((d, d))
    rho = A @ A.T
    rho /= np.trace(rho)
    Ms = rng.standard_normal((d, d))
    O0 = traceless((Ms + Ms.T) / 2, d)
    O0 = O0 / np.sqrt(np.trace(O0 @ O0))
    true_val = float(np.real(np.trace(O0 @ rho)))
    Ks_id = depolarizing(d, 1.0)
    Ohat_blind = (np.linalg.pinv(shadow_superop(Ks_id, d, W)).conj().T @ O0.reshape(d * d)).reshape(d, d)
    ps = [0.35, 0.5, 0.65, 0.8, 0.9, 1.0]
    SH = 400000
    bl_m, bl_ci, va_m, va_ci, betas = ([], [], [], [], [])
    for j, p in enumerate(ps):
        Ks = depolarizing(d, p)
        _, b = alpha_beta(Ks, d)
        b = float(np.real(b))
        betas.append(b)
        sm = run(rho, Ohat_blind.astype(complex), W, d, p, SH, np.random.default_rng(1700 + j))
        m, ci = mean_ci(sm)
        bl_m.append(m / true_val)
        bl_ci.append(ci / abs(true_val))
        Ohat = (np.linalg.pinv(shadow_superop(Ks, d, W)).conj().T @ O0.reshape(d * d)).reshape(d, d)
        sv = run(rho, Ohat.astype(complex), W, d, p, SH, np.random.default_rng(1800 + j))
        v, vci = var_ci(sv)
        va_m.append(v)
        va_ci.append(vci)
        print('  fig14: p=%.2f beta=%.3f  blind ratio %.4f+-%.4f (exact %.4f, resid %+.2f CI)   var %.3f+-%.3f (exact %.3f, resid %+.2f CI)' % (p, b, m / true_val, ci / abs(true_val), (b - 1) / (d - 1), (m / true_val - (b - 1) / (d - 1)) / max(ci / abs(true_val), 1e-15), v, vci, _mvO(O0, rho, b, d), (v - _mvO(O0, rho, b, d)) / max(vci, 1e-15)), flush=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    pp = np.linspace(0.0, 1.0, 100)
    bb = pp * d + 1 - pp
    a1.plot(pp, (bb - 1) / (d - 1), '-', c=C['th'], lw=1.3, label='exact $(\\beta-1)/(d-1)$')
    a1.errorbar(ps, bl_m, yerr=bl_ci, fmt='o', ms=4.5, c=C['O'], mfc='white', capsize=2.5, lw=1.1, label='measured $\\langle\\hat o_{\\rm blind}\\rangle/\\mathrm{Tr}(O_0\\rho)$')
    a1.axhline(1.0, ls=':', c='0.5', lw=1, label='unbiased')
    a1.set_xlabel('depolarizing strength $p$')
    a1.set_ylabel('estimator / true value')
    a1.set_title('Noise-blind post-processing', fontsize=9)
    a1.legend(fontsize=7, loc='upper left')
    bs = np.linspace(min(betas) * 0.97, d, 120)
    a2.plot(bs, [_mvO(O0, rho, float(b), d) for b in bs], '-', c=C['th'], lw=1.3, label='exact variance, Eq. (varO)')
    a2.errorbar(betas, va_m, yerr=va_ci, fmt='s', ms=4.5, c=C['O'], mfc='white', capsize=2.5, lw=1.1, label='measured single-shot variance')
    a2.set_xlabel('diagonal weight $\\beta$   ($\\beta=d$ noiseless)')
    a2.set_ylabel('single-shot variance')
    a2.set_title('Noise never improves the protocol', fontsize=9)
    a2.legend(fontsize=7)
    a2.set_yscale('log')
    save(fig, 'fig14_noise_blind_monotone')
    print('  fig14: variance rises by %.1fx as beta falls from %.2f to %.2f' % (va_m[0] / va_m[-1], betas[-1], betas[0]))

def fig_gram_spectrum():
    """Figure 10: eigenvalues of the order-3 Gram matrix G_3(d)."""
    ds = np.linspace(2.0, 6.0, 160)
    eig = np.array([np.linalg.eigvalsh(np.power(d, _E)) for d in ds])
    eig = np.clip(eig, 0.0001, None)
    fig, ax = plt.subplots(figsize=W1)
    for k in range(5):
        ax.semilogy(ds, eig[:, k], c=C['U'], lw=1.4, alpha=0.9)
    for k in range(5, 15):
        ax.semilogy(ds, eig[:, k], c=C['O'], lw=1.0, alpha=0.6)
    ax.axvline(2, ls=':', c=C['th'], lw=1)
    ax.plot([], [], c=C['U'], label='5 radical eigenvalues')
    ax.plot([], [], c=C['O'], label='10 commutant eigenvalues')
    ax.set_xlabel('$d$')
    ax.set_ylabel('eigenvalues of $G_3(d)$')
    ax.set_title('Collapse of the order-3 Gram spectrum at $d=2$')
    ax.set_ylim(0.001, None)
    ax.legend(loc='lower right')
    ax.annotate('$\\dim\\mathrm{rad}\\,B_3(2)=5$', xy=(2.02, 0.003), fontsize=8)
    save(fig, 'fig6_gram_spectrum')

def fig_amp_damping():
    """Not in the paper: amplitude damping, end to end."""
    from channels import amplitude_damping_1q, tensor_channel
    from mc_fast import run_trajectories
    rng = np.random.default_rng(9)
    d = 4
    p = 0.75
    Ks = tensor_channel([amplitude_damping_1q(p)] * 2)
    _, beta = alpha_beta(Ks, d)
    rho = random_state(d, rng)
    O = random_symmetric_obs(d, rng)
    true = np.real(np.trace(O @ rho))
    anvar = analytic_var_O(O, rho, beta, d)
    M = 40
    N = 250000
    E = run_trajectories(rho, O, Ks, d, beta, N, M, rng)
    step = np.unique(np.geomspace(50, N, 60).astype(int))
    cs = np.cumsum(E, axis=1)
    cs2 = np.cumsum(E ** 2, axis=1)
    ks = step.astype(float)
    rmt = cs[:, step - 1] / ks
    rvt = cs2[:, step - 1] / ks - rmt ** 2
    rm = rmt.mean(0)
    rse = rmt.std(0) / np.sqrt(M)
    rv = rvt.mean(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=W2)
    a1.axhline(true, c=C['th'], ls='--', lw=1, label='$\\mathrm{tr}(O\\rho)$')
    a1.fill_between(step, rm - 1.96 * rse, rm + 1.96 * rse, color=C['U'], alpha=0.25, label='95% CI ($M{=}40$ runs)')
    a1.plot(step, rm, c=C['U'], lw=1.3, label='running estimate (avg)')
    a1.set_xscale('log')
    a1.set_xlabel('shots $N$')
    a1.set_ylabel('$\\hat o$')
    a1.legend()
    a1.set_title('Amplitude damping: unbiasedness ($\\beta=(1{+}p)^n$)')
    a2.axhline(anvar, c=C['th'], ls='--', lw=1, label='analytic $\\mathrm{Var}_{\\mathbb{O}}$')
    a2.plot(step, rv, c=C['acc'], lw=1.3, label='empirical (avg, $M{=}40$)')
    a2.set_xscale('log')
    a2.set_xlabel('shots $N$')
    a2.set_ylabel('variance')
    a2.legend()
    a2.set_title('Variance formula (non-unital noise)')
    save(fig, 'fig7_amp_damping')

def fig_sample_complexity():
    """Figure 2: median-of-means failure probability."""
    from median_of_means import failure_rate_pooled
    from mc_depol import run_depol
    from mc_unitary import run_unitary
    from many_body import tfim, ground_state, traceless, var_O as mvO, var_U as mvU
    p = 0.9
    fig, a2 = plt.subplots(figsize=W1)
    n = 3
    dd = 2 ** n
    Ks = depolarizing(dd, p)
    _, bt = alpha_beta(Ks, dd)
    bt = float(np.real(bt))
    _, psi = ground_state(tfim(n, 1.0, 1.0))
    rho = np.outer(psi, psi)
    O0 = traceless(tfim(n, 1.0, 1.0), dd)
    O0 = O0 / np.sqrt(np.real(np.trace(O0 @ O0)))
    true = float(np.real(np.trace(O0 @ rho)))
    vO = mvO(O0, rho, bt, dd)
    vU = mvU(O0, rho, bt, dd)
    ep, dl, MM = (0.1, 0.05, 1)
    K = int(np.ceil(2 * np.log(2 * MM / dl)))
    NO = int(np.ceil(34 * vO / ep ** 2))
    NU = int(np.ceil(34 * vU / ep ** 2))
    POOL = 1500000
    REPS = 1200
    poolO = run_depol(rho, O0, dd, p, bt, POOL, 1, np.random.default_rng(21)).ravel()
    poolU = run_unitary(rho, O0, Ks, dd, bt, POOL, np.random.default_rng(22), depol_p=p)
    rng = np.random.default_rng(99)
    for lbl, pool, Nb, col in (('orthogonal', poolO, NO, C['O']), ('unitary', poolU, NU, C['U'])):
        grid = sorted({max(2, int(Nb * x)) for x in (0.008, 0.015, 0.025, 0.04, 0.06, 0.1, 0.18, 0.35, 0.7, 1.0)})
        xs = []
        ys = []
        for N in grid:
            fr, bad = failure_rate_pooled(pool, true, ep, N, K, REPS, rng)
            xs.append(N)
            ys.append(max(fr, 1.0 / (2 * REPS)))
            print(f'  fig8: {lbl:11s} N={N:6d} K={K}  failure {bad:4d}/{REPS} = {fr:.4f}')
        a2.loglog(xs, ys, 'o-', ms=3.5, c=col, mfc='white', lw=1, label=lbl)
        a2.axvline(Nb, ls=':', c=col, lw=1)
        sm = [N for N, y in zip(xs, ys) if y <= dl]
        if sm:
            print(f'  fig8: {lbl} smallest N meeting delta = {sm[0]}, bound N = {Nb}  -> looser by {Nb / sm[0]:.0f}x')
    a2.axhline(dl, c=C['th'], ls='--', lw=1)
    a2.annotate('$\\delta=0.05$', xy=(NO * 0.3, dl * 1.3), fontsize=7)
    a2.annotate('bound $N$\n(dotted)', xy=(NO * 0.55, 0.003), fontsize=6.5, ha='right', color='0.3')
    a2.set_ylim(1.0 / (3 * REPS), 0.7)
    a2.set_xlabel('batch size $N$  ($K=%d$ batches)' % K)
    a2.set_ylabel('measured $\\Pr[\\,|\\hat o-\\mathrm{tr}(O\\rho)|>\\varepsilon\\,]$')
    a2.set_title(f'Median-of-means, measured ($n={n}$, $\\varepsilon={ep}$)')
    a2.legend(fontsize=7, loc='lower left')
    save(fig, 'fig8_sample_complexity')

def _ghz(n):
    """GHZ state vector."""
    import numpy as np
    d = 2 ** n
    v = np.zeros(d)
    v[0] = v[-1] = 1 / np.sqrt(2)
    return v

def _varO(A0, rho, beta, d):
    """Orthogonal variance, Eq. (varO)."""
    t1 = np.real(np.trace(A0 @ A0))
    t2 = np.real(np.trace(rho @ A0 @ A0))
    t3 = np.real(np.trace(A0 @ rho))
    return (d - 1) * (d + 2) / ((d + 4) * (beta - 1)) * ((d * (d + 3) - 4 * beta) / (2 * d * (beta - 1)) * t1 + 2 * t2) - t3 ** 2

def _varU(A0, rho, beta, d):
    """Unitary variance, Eq. (varU)."""
    t1 = np.real(np.trace(A0 @ A0))
    t2 = np.real(np.trace(rho @ A0 @ A0))
    t3 = np.real(np.trace(A0 @ rho))
    return (d * d - 1) / ((d + 2) * (beta - 1)) * ((d + d * d - 2 * beta) / (d * (beta - 1)) * t1 + 2 * t2) - t3 ** 2

def fig_complex_noise():
    """Figure 7: reality-tuned crossover under noise."""
    import scipy.linalg as sla
    from scipy.stats import unitary_group
    from exact_tools import exact_mean_var
    from channels import depolarizing, alpha_beta
    rng = np.random.default_rng(1)
    d = 4
    p = 0.85
    Ks = depolarizing(d, p)
    _, beta = alpha_beta(Ks, d)
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = G @ np.conj(G.T)
    rho /= np.trace(rho)
    Ms = rng.standard_normal((d, d))
    O = (Ms + Ms.T) / 2
    O0 = O - np.trace(O) * np.eye(d) / d
    Hh = unitary_group.rvs(d, random_state=rng)
    Hh = (Hh + np.conj(Hh.T)) / 2

    def reality(W):
        return float(np.real(sum((abs(np.conj(W[:, k]) @ np.conj(W[:, k])) ** 2 for k in range(d)))) / d)

    def paired(t):
        B = np.array([[np.cos(t), 1j * np.sin(t)], [1j * np.sin(t), np.cos(t)]])
        W = np.zeros((d, d), dtype=complex)
        W[:2, :2] = B
        W[2:, 2:] = B
        return W

    def varx(W):
        return float(np.real(exact_mean_var(O, rho, Ks, d, W)[1]))
    ts = np.linspace(0, np.pi / 4, 41)
    rea = np.array([reality(paired(t)) for t in ts])
    vs = np.array([varx(paired(t)) for t in ts])
    idx = np.argsort(rea)
    rea = rea[idx]
    vs = vs[idx]
    chk = [(reality(sla.expm(1j * th * Hh)), varx(sla.expm(1j * th * Hh))) for th in (0.33, 0.65, 0.98, 1.3)]
    chk_r = [c[0] for c in chk]
    chk_v = [c[1] for c in chk]
    from mc_general import run, var_ci
    from exact_tools import shadow_superop
    SH = 3000000
    mc_r = []
    mc_v = []
    mc_ci = []
    for j, sg in enumerate([0.85, 0.5, 0.22, 0.0]):
        W = paired(np.arccos(np.sqrt(sg)) / 2)
        mc_r.append(reality(W))
        Oh = (np.linalg.pinv(shadow_superop(Ks, d, W)).conj().T @ O.reshape(d * d)).reshape(d, d)
        sm = run(rho, Oh, W, d, p, SH, np.random.default_rng(500 + j))
        v, ci = var_ci(sm)
        mc_v.append(v)
        mc_ci.append(ci)
    lo = _varO(O0, rho, beta, d)
    hi = _varU(O0, rho, beta, d)
    from scipy.optimize import brentq
    tx = brentq(lambda t: varx(paired(t)) - hi, 0.0, np.pi / 4, xtol=1e-10)
    sx = reality(paired(tx))
    fig, ax = plt.subplots(figsize=W1)
    ax.axhline(lo, ls='--', c=C['O'], lw=1, label='real basis (factor-2 optimal)')
    ax.axhline(hi, ls='--', c=C['U'], lw=1, label='unitary reference')
    ax.plot(rea, vs, '-', c=C['th'], lw=1.3, label='exact variance (complex basis)')
    ax.errorbar(mc_r, mc_v, yerr=mc_ci, fmt='o', ms=3.5, c=C['th'], mfc='white', capsize=2, lw=1, label='Monte Carlo (95% CI)')
    ax.plot(chk_r, chk_v, '+', ms=5, c=C['acc'], mew=1.1, label='unrelated basis family')
    ax.plot([sx], [hi], marker='o', ms=4, c=C['U'], zorder=5)
    ax.annotate('crosses at $\\varsigma=%.2f$' % sx, xy=(sx, hi), xytext=(sx - 0.3, hi + 1.55), fontsize=7.5, color=C['U'], arrowprops=dict(arrowstyle='-', lw=0.7, color=C['U'], shrinkA=0, shrinkB=2))
    ax.set_xlabel('reality fraction $\\varsigma=\\alpha_{\\mathrm{r}}/d$  (1: real $\\;\\to\\;$ 0: fully complex)')
    ax.set_ylabel('single-shot variance')
    ax.invert_xaxis()
    ax.set_title('Complex-basis crossover under noise ($n=2$, depol $p=0.85$)')
    ax.legend(fontsize=7.5, loc='upper left')
    save(fig, 'fig10_complex_noise')
    print(f'  fig10: lo={lo:.4f} hi={hi:.4f} varsigma=0 -> {vs[0]:.4f} ({vs[0] / lo:.3f}x real, {vs[0] / hi:.3f}x unitary), crossing at varsigma={sx:.4f}')
    print(f'  fig10: varsigma-only check, max |dVar| vs paired family = {max((abs(v - varx(paired(np.arccos(np.sqrt(r)) / 2))) for r, v in chk)):.2e}')
    for r, v, c in zip(mc_r, mc_v, mc_ci):
        print(f'  fig10: MC varsigma={r:.3f}  {v:.4f} +- {c:.4f}   exact {varx(paired(np.arccos(np.sqrt(max(r, 0))) / 2)):.4f}')

def fig_ghz():
    """Figure 8: the advantage as a function of kappa; GHZ case study."""
    from instance_ensembles import kappa_scatter, collapse_residual, ghz
    from many_body import tfim, ground_state, traceless, beta_depol, kappa_exact, rho_L, rho_S, second_moments
    p = 0.9
    rows = [r for r in kappa_scatter(ns=(2, 4, 6), p=p, per_family=15, seed=1) if r[3] in ('GHZ fidelity', 'TFIM energy', 'dense symmetric', 'unaligned projector')]
    resid = collapse_residual(rows, p=p)
    STY = {'GHZ fidelity': ('*', C['U'], 8, 'GHZ fidelity'), 'TFIM energy': ('D', C['O'], 4, 'TFIM energy'), 'dense symmetric': ('^', C['acc'], 4, 'dense symmetric'), 'unaligned projector': ('o', '0.45', 3.4, 'projector, unaligned')}
    fig = plt.figure(figsize=(6.5, 3.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0], hspace=0.08, wspace=0.3)
    a1 = fig.add_subplot(gs[0, 0])
    ar = fig.add_subplot(gs[1, 0], sharex=a1)
    a2 = fig.add_subplot(gs[:, 1])
    kk = np.logspace(-0.7, 3.0, 400)
    for d, ls in zip(sorted({r[2] for r in rows}), [(0, (1, 1.4)), (0, (4, 1.6)), '-']):
        b = beta_depol(d, p)
        a1.semilogx(kk, (rho_L(d, b) * kk + rho_S(d, b)) / (kk + 1), c='0.55', lw=0.8, ls=ls, zorder=1)
    a1.annotate('exact identity: $n=2,4,6$', xy=(1.6, 1.3), fontsize=6.5, color='0.35')
    a1.semilogx(kk, (2 * kk + 1) / (kk + 1), c=C['th'], lw=1.2, ls='--', zorder=2)
    a1.annotate('$\\frac{2\\kappa+1}{\\kappa+1}$ (large $d$)', xy=(60, 1.62), fontsize=7, color=C['th'])
    for fam, (mk, col, ms, lab) in STY.items():
        xs = [r[0] for r in rows if r[3] == fam]
        ys = [r[1] for r in rows if r[3] == fam]
        if not xs:
            continue
        a1.semilogx(xs, ys, mk, c=col, ms=ms, alpha=0.95, mew=0.9, mfc='none' if mk in 'o^D' else col, label=lab, zorder=3)
    a1.set_ylabel('$\\mathbb{E}_{\\mathbb{U}}[\\hat o^2]/\\mathbb{E}_{\\mathbb{O}}[\\hat o^2]$')
    a1.set_ylim(1.0, 2.15)
    a1.tick_params(labelbottom=False)
    a1.legend(fontsize=6.8, loc='lower right', ncol=2, columnspacing=0.8, handletextpad=0.3)
    rs = []
    for k, rat, d, fam, _np in rows:
        b = beta_depol(d, p)
        ex = (rho_L(d, b) * k + rho_S(d, b)) / (k + 1)
        rs.append((k, abs(rat - ex) / ex))
    ar.loglog([r[0] for r in rs], [max(r[1], 1e-17) for r in rs], '.', c='0.35', ms=2.4)
    ar.set_ylim(1e-17, 1e-13)
    ar.set_yticks([1e-16, 1e-14])
    ar.set_ylabel('rel. dev.', fontsize=7)
    ar.tick_params(labelsize=6.5)
    ar.set_xlabel('$\\kappa=\\dfrac{(d^2+3d-4\\beta)\\,\\mathrm{tr}(O_0^2)}{4d(\\beta-1)\\,\\mathrm{tr}(\\rho O_0^2)}$')
    ns = list(range(2, 11))
    kg = []
    kt = []
    kr = []
    for n in ns:
        d = 2 ** n
        b = beta_depol(d, p)
        g = ghz(n)
        Og = traceless(np.outer(g, g), d)
        kg.append(kappa_exact(Og, np.outer(g, g), b, d))
        H = tfim(n, 1.0, 1.0)
        _, ps = ground_state(H)
        rt = np.outer(ps, ps)
        kt.append(kappa_exact(traceless(H, d), rt, b, d))
        rr = np.random.default_rng(n)
        Ms = rr.standard_normal((d, d))
        kr.append(kappa_exact(traceless((Ms + Ms.T) / 2, d), rt, b, d))
    a2.semilogy(ns, kr, '^-', c=C['acc'], ms=4, label='dense symmetric ($\\kappa\\sim d$)')
    a2.semilogy(ns, kt, 'D-', c=C['O'], ms=4, label='TFIM energy ($\\kappa\\sim d/n$)')
    a2.semilogy(ns, kg, '*-', c=C['U'], ms=7, label='GHZ fidelity ($\\kappa\\to\\frac{1}{4p}$)')
    a2.axhline(1 / (4 * p), ls=':', c=C['U'], lw=1)
    a2.annotate('$\\frac{1}{4p}$', xy=(9.3, 1 / (4 * p) * 1.15), color=C['U'], fontsize=8)
    a2.set_xlabel('qubits $n$')
    a2.set_ylabel('$\\kappa$')
    a2.legend(fontsize=7, loc='upper left')
    save(fig, 'fig9_ghz_case_study', tight=False)
    fig = None
    ks = np.array([r[0] for r in rows])
    print(f'  fig9: {len(rows)} instances, kappa {ks.min():.3g}..{ks.max():.3g} ({np.log10(ks.max() / ks.min()):.1f} decades), collapse residual {resid:.2e}')
    print(f'  fig9: rho_L(d=4)={rho_L(4, beta_depol(4, p)):.4f} (>2, so the large-d curve is not an envelope at d=4)')

def fig_chirality():
    """Figure 9: chirality, an observable outside the real visible space."""
    import scipy.linalg as sla
    from scipy.stats import unitary_group
    from many_body import heisenberg_j1j2, chirality, ground_state, traceless
    from exact_tools import exact_mean_var, shadow_superop
    from channels import depolarizing, alpha_beta
    from mc_general import run, mean_ci, var_ci
    SHOTS = 2000000
    n = 3
    d = 2 ** n
    p = 0.9
    Ks = depolarizing(d, p)
    _, beta = alpha_beta(Ks, d)
    chi = chirality(n, 0, 1, 2)
    H = heisenberg_j1j2(n, 1.0, 0.5, pbc=True)
    _, psic = ground_state(H + 0.8 * chi)
    rho = np.outer(psic.conj(), psic)
    O_en = traceless(np.real(H), d)
    true_chi = np.real(np.trace(rho @ chi))
    rng = np.random.default_rng(7)
    Hh = unitary_group.rvs(d, random_state=rng)
    Hh = (Hh + Hh.conj().T) / 2
    rl = lambda Wm: np.real(sum((abs(np.sum(Wm[:, k].conj() * Wm[:, k].conj())) ** 2 for k in range(d)))) / d
    ths = np.linspace(0.0, 1.6, 20)
    ar = []
    vchi = []
    ven = []
    mchi = []
    for th in ths:
        Wm = np.eye(d) if th == 0 else sla.expm(1j * th * Hh)
        ar.append(rl(Wm))
        m, v = exact_mean_var(chi, rho, Ks, d, Wm)
        mchi.append(np.real(m))
        vchi.append(np.real(v))
        _, v2 = exact_mean_var(O_en, rho, Ks, d, Wm)
        ven.append(np.real(v2))
    ar = np.array(ar)
    vchi = np.array(vchi)
    ven = np.array(ven)
    mchi = np.array(mchi)
    ok = ar < 0.999
    mth = [0.45, 0.7, 0.95, 1.2, 1.5]
    q = {k: [] for k in ('ar', 'm', 'mci', 'v', 'vci', 've', 'veci', 'ex_m', 'ex_v', 'ex_ve')}
    for j, th in enumerate(mth):
        Wm = sla.expm(1j * th * Hh)
        q['ar'].append(rl(Wm))
        Si = np.linalg.pinv(shadow_superop(Ks, d, Wm))
        Oc = (Si.conj().T @ chi.reshape(d * d)).reshape(d, d)
        Oe = (Si.conj().T @ O_en.reshape(d * d)).reshape(d, d)
        sc = run(rho, Oc, Wm, d, p, SHOTS, np.random.default_rng(1000 + j))
        m, ci = mean_ci(sc)
        v, vci = var_ci(sc)
        se = run(rho, Oe, Wm, d, p, SHOTS, np.random.default_rng(2000 + j))
        ve, veci = var_ci(se)
        mx, vx = exact_mean_var(chi, rho, Ks, d, Wm)
        _, vex = exact_mean_var(O_en, rho, Ks, d, Wm)
        for k, val in zip(('m', 'mci', 'v', 'vci', 've', 'veci', 'ex_m', 'ex_v', 'ex_ve'), (m, ci, v, vci, ve, veci, np.real(mx), np.real(vx), np.real(vex))):
            q[k].append(val)
    Q = {k: np.array(v) for k, v in q.items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.75))
    a1.axhline(true_chi, ls='--', c=C['th'], lw=1, label='true $\\mathrm{tr}(\\chi\\rho)$')
    a1.axhline(0.0, c='0.8', lw=0.8)
    a1.plot(ar[ok], mchi[ok], '-', c=C['O'], lw=1.3, label='exact $\\mathbb{E}[\\hat\\chi]$')
    a1.plot(Q['ar'], Q['m'], 'o', c=C['O'], ms=4, mfc='white', mew=1.1, label='Monte Carlo')
    a1.plot([1.0], [0.0], 'X', c=C['U'], ms=9, zorder=5)
    a1.annotate('real basis: ' + '$\\chi$ in kernel', xy=(1.0, 0.0), xytext=(0.5, 0.11), color=C['U'], fontsize=6.8, ha='left', arrowprops=dict(arrowstyle='->', color=C['U'], lw=0.9, shrinkA=0, shrinkB=4))
    a1.set_ylim(-0.06, 0.56)
    a1.set_ylabel('$\\mathbb{E}[\\hat\\chi]$')
    a1.set_title('Chirality $\\chi=\\mathbf{S}_i\\!\\cdot\\!(\\mathbf{S}_j\\!\\times\\!\\mathbf{S}_k)$')
    a1.legend(loc='center left', fontsize=6.8)
    a1.set_xlabel('basis reality $\\alpha_{\\mathrm{r}}/d$')
    a2.plot(ar[ok], vchi[ok], '-', c=C['O'], lw=1.3, label='exact $\\mathrm{Var}[\\hat\\chi]$')
    a2.plot(Q['ar'], Q['v'], 'o', c=C['O'], ms=4, mfc='white', mew=1.1)
    a2.plot(ar, ven, '-', c=C['acc'], lw=1.3, label='exact $\\mathrm{Var}[\\hat H_0]$')
    a2.plot(Q['ar'], Q['ve'], 's', c=C['acc'], ms=4, mfc='white', mew=1.1)
    a2.axvline(1.0, ls=':', c=C['U'], lw=1.1)
    a2.set_yscale('log')
    a2.annotate('$\\chi$ unestimable', xy=(1.0, 20), xytext=(0.46, 60), color=C['U'], fontsize=6.8, ha='left', arrowprops=dict(arrowstyle='->', color=C['U'], lw=0.9, shrinkA=0, shrinkB=2))
    a2.set_ylabel('single-shot variance')
    a2.set_title('Cost of the enlarged visible space')
    a2.legend(loc='upper left', fontsize=6.8)
    a2.set_xlabel('basis reality $\\alpha_{\\mathrm{r}}/d$')
    save(fig, 'fig12_chirality')
    print('   residuals (CI units): mean %.2f  var(chi) %.2f  var(H) %.2f' % (np.abs((Q['m'] - Q['ex_m']) / Q['mci']).max(), np.abs((Q['v'] - Q['ex_v']) / Q['vci']).max(), np.abs((Q['ve'] - Q['ex_ve']) / Q['veci']).max()))
if __name__ == '__main__':
    fig_variance_ratio()
    fig_local_advantage()
    fig_noise_dependence()
    fig_complex_crossover()
    fig_convergence()
    fig_gram_spectrum()
    fig_amp_damping()
    fig_sample_complexity()
    fig_ghz()
    fig_chirality()
    fig_complex_noise()
    fig_noise_blind_monotone()
    import os as _os
    print('all figures written to', _os.path.abspath(OUT))
