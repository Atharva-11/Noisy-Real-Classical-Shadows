# Which code makes which figure

The figure numbers in the left column are the numbers **as printed in the paper**. 

Every figure is produced by a function in [`code/make_figures.py`](code/make_figures.py). Running

```
cd code && python make_figures.py
```

writes all of them to `figures/` as both `.pdf` and `.png`.

| Fig. | file | generator | modules it uses |
|---|---|---|---|
| 1 | `fig1_variance_ratio.pdf` | `fig_variance_ratio()` | `instance_ensembles`, `many_body`, `mc_depol`, `mc_unitary` |
| 2 | `fig8_sample_complexity.pdf` | `fig_sample_complexity()` | `many_body`, `mc_depol`, `mc_unitary`, `median_of_means` |
| 3 | `fig14_noise_blind_monotone.pdf` | `fig_noise_blind_monotone()` | `channels`, `exact_tools`, `many_body`, `mc_general` |
| 4 | `fig4_noise_dependence.pdf` | `fig_noise_dependence()` | `noise_zoo` |
| 5 | `fig3_local_advantage.pdf` | `fig_local_advantage()` | `many_body`, `mc_local` |
| 6 | `fig5_complex_crossover.pdf` | `fig_complex_crossover()` | `complex_third_moment`, `many_body`, `mc_general` |
| 7 | `fig10_complex_noise.pdf` | `fig_complex_noise()` | `channels`, `exact_tools`, `mc_general` |
| 8 | `fig9_ghz_case_study.pdf` | `fig_ghz()` | `instance_ensembles`, `many_body` |
| 9 | `fig12_chirality.pdf` | `fig_chirality()` | `channels`, `exact_tools`, `many_body`, `mc_general` |
| 10 | `fig6_gram_spectrum.pdf` | `fig_gram_spectrum()` | none directly — see the note below |

## What each figure shows

| Fig. | subject |
|---|---|
| 1 | Exact variance ratio `Var_𝕌 / Var_𝕆` under depolarizing noise (`p = 0.9`), over 500 random `(ρ, O)` instances per point. |
| 2 | The median-of-means estimator, run: measured failure probability against batch size. |
| 3 | The noise-blind protocol, and the monotonicity of the variance in the noise. |
| 4 | The depolarizing parameter `f(ℰ) = 2(β−1)/((d−1)(d+2))` for all five noise models, against the noiseless value `2/(d+2)`. |
| 5 | Local Pauli shadow seminorm under single-qubit depolarizing noise: the orthogonal `(2f₁²)^{-wt}` against the unitary counterpart. |
| 6 | Complex-basis crossover against the reality fraction `ς`, at `n = 3, 5, 7`, two panels, parameter and variance. |
| 7 | Reality-tuned crossover under noise at `n = 2`: the exact single-shot variance as `ς` falls from 1 to 0. |
| 8 | The advantage as a function of the single parameter `κ`, including the GHZ case study. |
| 9 | An observable outside the real visible space: the three-spin chirality. |
| 10 | The fifteen eigenvalues of the order-three Gram matrix `G₃(d) = (d^{ℓ(i,j)})`, five of which vanish as `d → 2`. |

## The Gram-matrix code

Figure 10 is drawn from the loop-exponent matrix `ℓ(i,j)`, which is held as an integer constant
`_E` inside `make_figures.py`. The matrix itself is *constructed* in:

- [`code/weingarten_core.py`](code/weingarten_core.py): `gram_k3(d)` builds `G₃(d)` from the
  fifteen Brauer diagram operators, and `basis_k3` / `perm_operator` / `omega_operator` build the
  operators.
- [`code/weingarten_full.py`](code/weingarten_full.py): the full symbolic derivation: `G₂` and its
  inverse `Wg₂`, the exponent matrix, `det G₃(d)` and its factorisation, and the five radical
  generators of `𝔅₃(2)`.
- [`code/twirl_engine.py`](code/twirl_engine.py): the exact `𝕆(d)` twirl at `k = 2, 3` as the
  orthogonal projector onto the commutant.

`_E` and `gram_k3` agree exactly.

[`code/two_design_gap.py`](code/two_design_gap.py) supports the optimality paragraph of §8. Run it directly (`python two_design_gap.py`) and it prints every number that paragraph quotes: the group orders 288 and 1152, the commutant dimensions (3, 21) against (3, 15), the agreement of the two shadow channels, and the 20.6 % second-moment gap.

## Files not used in the paper

`make_figures.py` also contains `fig_convergence()` (`fig2_convergence`) and `fig_amp_damping()`
(`fig7_amp_damping`). Both run, and their dependencies are shipped, but neither figure appears in
the paper.
