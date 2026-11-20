# Zachary Diaz — Sep–Oct 2026 runs

Updated: 2026-11-05

Zachary Diaz took and analysed the Sep–Oct 2026 0.5 m NaCl temperature
trials (`data/20260917`, `20260923`, `20260924`, `20260925`) and the
2026-10-09 5 wt% NaCl σ(P) run (`data/20261009`).

The analysis, import and exclusions are documented in
`docs/DIAZ_2026_ANALYSIS.md`; the scripts are `import_zanalysis_workbook.py`
and `curate_diaz_2026.py` at the repo root; his workbooks are in `labnotes/`.

Not yet merged into the Cortes compilation. Candidate overlap: 0.5 m NaCl
(the Cortes Aug 2025 high-pressure series) at 1 bar, -1 to 25 C, assessed at
P = 0.1 MPa. Open items (Oct 9 cell constant, the 'Lit' source, the
trial-to-trial spread) are in `coordination/open-questions.md`.

## 8 S/m KCl standard sigma(P) — Oct 14–15 and Nov 5 2026

Both vessel runs are the 8 S/m KCl standard throughout (Steve, 2026-11-05).
`plot_standard_sigma_P.py` fits every sweep and plots them alongside the
Cortes 0.5 m NaCl series (`standard_sigma_P_vs_cortes.pdf`;
per-sweep values in `diaz_standard_sigma_P.csv`). K is set by the first
three sweeps at ~1 MPa in each run: 122.56 (Oct 14), 120.54 (Nov 5).

| P (MPa) | Oct 14–15, 296 K | Nov 5, 298 K |
|---|---|---|
| ~1 | 8.000 (ref) | 8.000 (ref) |
| 200 | 8.552 (+6.9%) | 8.623 (+7.8%) |
| back to ~2 / 76 | 7.929 at 2 MPa (−0.9%) | 8.369 at 76 MPa (ascent 8.345) |

### Unreliable sweeps (kept, under investigation)

20 sweeps have > 5% fit uncertainty in R (`noisy` in the CSV). Two kinds:

- **Fit unsure, sigma on the curve:** Oct 14 at 100 MPa and Oct 15 at
  52–100 MPa (≈5% uncertainty, sigma within 0.15% of neighbours).
- **Fit off the curve:** Oct 14 at 124 MPa (+6.6%, three sweeps) and Nov 5
  descent at 177 MPa (2 of 3) and 152 MPa (3) (+2.3%, ≈13% uncertainty).

The raw spectra do not show the excursion. Taking R as |Z| at the
phase-zero frequency (no circuit model) puts every one of these sweeps on
the smooth curve within 0.15%, and the raw |Z| and phase at 1 MHz–100 Hz
change smoothly from the neighbouring good sweeps. So the jumps are the
CPE fit converging to a different minimum, not a change in the solution
or the cell.

**Confirmed by refit (2026-11-05, `refit_chained.py`).** Starting each
sweep's fit from the previous sweep's fitted parameters removes every
excursion: all 20 flagged sweeps drop below 5% R uncertainty, and the
chained σ matches the phase-zero σ within 0.11% (Oct 14–15) and 0.06%
(Nov 5) for every sweep, against 6.7% and 2.3% for the production fit.
The residual is the same or lower (median 0.293 vs 0.295 Ω), so the
production fit had stopped in a worse local minimum. Neither the
data nor the cell changed. Figure: `refit_chained.pdf`; values:
`refit_chained.csv`. The production fitter (`Solution.fit_circuit`) still
starts every sweep from the same fixed guess; changing it is a separate
decision.

Note also that the Oct 15 return to 2 MPa reads 0.9% below the Oct 14
start, while Nov 5 retraces within 0.3% at 76–126 MPa.

### Exclusions (Steve, 2026-11-05)

Excluded in the zAnalysis configs (rows kept, `exclude = x`, reason in notes):

| Run | Sweeps | Production σ vs curve |
|---|---|---|
| Nov 5 | 14:25, 14:26 (177 MPa); 14:49–14:51 (152 MPa) | +2.3% |
| Oct 14 | 13:37–13:39 (124 MPa) | +6.6% |

These are the only off-trend points in either run. In both cases the raw
spectrum stays on the curve (phase-zero |Z|), and a chained-guess refit
removes the jump. So the "spontaneous" excursions come from the fit, not
the system. The 12 other Oct 14–15 sweeps with ~5% fit uncertainty sit on
the curve and are kept. `data/20261014` and `20261015` now have configs
(KCl; K from the first three Oct 14 sweeps, which also calibrate Oct 15).
