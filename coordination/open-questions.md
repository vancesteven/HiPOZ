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
