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
or the cell. Next: refit with the previous sweep's parameters as the
initial guess, or with basin hopping, and confirm the excursions vanish.

Note also that the Oct 15 return to 2 MPa reads 0.9% below the Oct 14
start, while Nov 5 retraces within 0.3% at 76–126 MPa.
