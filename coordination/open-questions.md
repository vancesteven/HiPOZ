# Open questions — hipozgenai

Anything ambiguous, destructive, or scientifically consequential goes here and
work **stops** until Steve resolves it. Include what you were doing, what the
options are, and your recommendation.

## 2026-09-15 — uncommitted working-tree changes excluded from PR #2

While propagating hipozgenai to main (PR #2), the working tree held changes
that have never been committed, so they are not in the PR:

- Modified: 20 `cortes2026/cortes_plots/*.pdf` (regenerated plots),
  `tests/test_gui_reorganization.py`, `.gitignore`
- Untracked: root-level planning notes (`PLOTTING_ARCHITECTURE.md`,
  `CLEANUP_EXECUTED.md`, etc.), `JesusCortes/`, `data/2025081[8-20]Cortes/`,
  `cortes2026/Cortes2026BenchtopData_pre20250624.csv`, mahboub2026 drafts
  (`main_revisions2.tex`, `supplement.tex`, `generate_correct_csv.py`, plus a
  `.py.save` backup and an `..._INCORRECT.csv`), several `tests/test_*.py`,
  `plot_all.sh`, `CLAUDE.md`, `AGENTS.md`, `plans/CODEX-QUEUE.md`,
  `coordination/inbox/`, this file
- Options: (a) commit the keepers (Aug 2025 Cortes data, new tests, agent-lane
  files) and delete the junk (`.py.save`, `_INCORRECT.csv`); (b) leave all
  as-is; (c) itemized review with Steve.
- Recommendation: (c) — the mix includes lab data and manuscript drafts, which
  are scientifically consequential and not mine to triage.

## 2026-09-15 — KCl reference value mismatch in tests/test_benchtop_data.py

`test_kcl_extraction` hard-codes an expected conductivity of 55.88 mS/cm for
20C, 0.5 M KCl ("from row 8, column 7"), but `JesusData2025.csv` currently
holds 57.83 mS/cm in that cell, so the test fails (tolerance 0.1). Either the
CSV was revised after the test was written or the test's expected value is
wrong. This is a scientific-data question, so neither the expected value nor
the tolerance was changed; the test is committed as-is and fails until
resolved. The companion NaCl:MgSO4 check (expected 69.09) passes.

Steve: which number is correct for 20C, 0.5 M KCl — 55.88 or 57.83?

## 2026-10-03 — what are the Sept 22-25 "Default" runs? (affects sigma(P,T) map)

The 171 raw sweeps uploaded to main in data/20260922-20260925 (all at 1-4 MPa,
~295-296 K, description "Default") were circuit-fitted with the usual CPE
model and converted with the curated dataset's shared K_cell = 128.40 1/m.
The resulting provisional sigma values step between flat plateaus (e.g. 9/22:
5.9 -> 6.6 -> 7.9 -> 8.6 S/m; 9/25 ends 7.5 -> 9.1 -> 11.7 -> 13.6), i.e.
these are NOT one fixed 52.168-ppt standard — they look like deliberate
composition/condition steps. Only ~9% of sweeps fall near nominal 8 S/m.

Steve: what was varied in these runs (dilutions? different standards? cell
work?), and which sweeps, if any, are the 8 S/m standard? Until then they are
excluded from quantitative pressure-trend fits; the plateaus near 8.3 S/m at
1-3 MPa vs the curated 8.11 S/m at 190 MPa (same T band) would imply only a
~ -2% change over 190 MPa if they are the undiluted standard with unchanged
K_cell.

## 2026-10-05 — regenerate Cortes McCleskey figures with fixed I + speciation?

Two model corrections landed on hipozgenai (see audit, 2026-10-05):
1. elecCondMcCleskey2012 now evaluates lambda_i at the TOTAL effective ionic
   strength, as MC12 specifies (was per-ion). KCl standards: errors improve
   from +2.3..+4.6% to within +-0.75%. This shifts EVERY McCleskey curve.
2. The Cortes pipeline (cortes_mccleskey.py, plot_cortes_with_mccleskey.py)
   now supports WATEQ4F speciation ('auto': on when Reaktoro is present).
   Benchtop validation shows MgSO4 RMS error is 78.6% WITHOUT speciation.

Both change figures intended for the Cortes et al. (2026) paper. Steve:
confirm before the paper figures are regenerated with the corrected model
(run on the Mac, where Reaktoro is installed, so 'auto' speciation engages).
Also: extend speciation.VERIFIED_COMPOUNDS to Na2SO4/Na2CO3 after
validate_speciation.py's MC12+spec column checks out there — the lambda
coefficients for NaSO4-/KSO4-/NaCO3- are now verified verbatim against the
published USGS table, so the old "provisional" doubt is retired.

## 2026-10-08 — two gaps left after the Cortes figure regeneration

1. **Mixtures are not speciated.** WATEQ4F speciation is only implemented for
   single salts (speciation.SALT_RECIPES). The NaCl:MgSO4 and Na2SO4:KCl
   mixture figures therefore get the ionic-strength fix but still assume full
   dissociation. For NaCl:MgSO4 this overestimates the model, because
   MgSO4(aq) association is large (only 26-59% of Mg stays free in pure
   MgSO4 over 0.02-1.7 m). Recommendation: generalize free_ion_molalities()
   to take an arbitrary {ion: molality} input with the union of elements
   (e.g. 'Na Mg Cl S O H'). It's a small change in Reaktoro, but it changes
   the mixture figures, so it needs your OK and has to be verified on the Mac.
2. **Four eis_plots weren't rebuilt and still show the pre-fix model:**
   mccleskey_comparison/MgSO4_vs_{concentration,temperature}_mccleskey.pdf,
   mixtures/Mixtures_vs_concentration.pdf, and
   single_salts/NaCl_vs_concentration.pdf. They come from the April
   0c435df run. The current plot_cortes_with_mccleskey.py run on the default
   dates (20250813-15Cortes) doesn't produce them, apparently because those
   dates have no MgSO4 single-salt EIS data. Copies are already in the
   archive. Options: (a) regenerate them using the dates that originally
   produced them, if you know which ones; (b) remove them from eis_plots/
   so nobody mistakes them for current results (git rm; the archive keeps a
   copy); (c) leave them. Recommendation: (b), unless they are paper figures.
