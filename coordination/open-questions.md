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
