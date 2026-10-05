# Status — hipozgenai

Updated: 2026-10-05

Refresh the `Updated:` line and the affected sections in any session that
pushes commits, integrates artifacts, or changes a queue.

## Current focus

Cortes 2026 paper: McCleskey/WATEQ4F model corrections — see audit 2026-10-05.
Previous: New sigma(P,T) GUI tab for diagnosing KCl-standard pressure trends; Sept 17-25
lab data merged in from origin/main. PR #3 (still open) now carries all of it.

## In flight

| Item | Owner | Status | Artifact |
|---|---|---|---|
| Merge origin/main lab data into hipozgenai | claude-code | verified | merge commit `eab7098`; audit entry 2026-09-14 in `coordination/audit.md` |
| Propagate hipozgenai to main | claude-code | verified | PR #2 merged by Steve 2026-09-15; merge commit `351853f` on `origin/main` contains branch tip `fd3718d` (`git merge-base --is-ancestor` confirmed) |
| Curate uncommitted working tree per Steve's decisions | claude-code | verified | commits `978540a` (tests/infra/docs), `e7b5f6f` (Aug 2025 Cortes + JesusCortes data), `af53f12` (mahboub2026 drafts); audit entry 2026-09-15 |
| New test suites pass in full project env (TeX + GUI) | claude-code | implemented, unverified | 20/23 collected tests pass in the Linux container; `test_plot_export`, `test_multi_panel_plot` need LaTeX and the two GUI suites need libEGL — run `pytest tests/` on macOS |
| sigma(P,T) 2D map plot function (gamryPlots.plot_conductivity_PT) | claude-code | verified | commit `1f33a65`; tests/test_pt_map.py (8 passing); real-data map from 13 KCl dates sent to Steve 2026-09-15 |
| sigma(P,T) GUI tab in DataSelector | claude-code | implemented, unverified | commit `1f33a65`; container lacks libEGL - launch GUI on macOS and view the new tab |
| Merge origin/main Sept 17-25 data uploads | claude-code | verified | merge commit `58906cd`; `git merge-base --is-ancestor origin/main HEAD` passes; 248 files under data/20260917-20260925 |
| plot_sigma_PT.py CLI (map from curated/raw data) | claude-code | verified | tests/test_plot_sigma_PT.py (4 passing); full KCl run reproduces delivered figures in sigma_PT_plots/ |
| 3D sigma(P,T) surface (function + CLI output) | claude-code | verified | tests/test_pt_map.py 15/15 passing; sigma_PT_surface.png delivered 2026-10-03 |
| 3D sigma(P,T) GUI tab ("σ(P,T) 3D", drag-to-rotate) | claude-code | implemented, unverified | needs GUI launch on macOS (no libEGL in container) |
| sigma vs P isotherms (function + CLI output) | claude-code | verified | tests/test_pt_map.py 19/19; sigma_P_isotherms.png delivered 2026-10-03 |
| sigma vs P isotherms GUI tab ("σ(P) isotherms") | claude-code | implemented, unverified | needs GUI launch on macOS (no libEGL in container) |
| McCleskey total-ionic-strength fix | claude-code | verified | tests/test_mccleskey_ionic_strength.py (KCl standards within 1%); audit 2026-10-05 |
| WATEQ4F speciation in Cortes pipeline (auto mode) | claude-code | implemented, unverified | fallback path tested here; speciated path needs Reaktoro (Mac) — run validate_speciation.py and pytest tests/test_speciation.py |
| Cortes paper figures regenerated with corrected model | claude-code | not implemented | approved by Steve 2026-10-05; old figures archived; run ./regenerate_cortes_figures.sh on the Mac (needs TeX + Reaktoro) |

Status must be one of `verified` / `implemented, unverified` / `not implemented`.

## Blockers

- `test_kcl_extraction` fails on a reference-value question (55.88 vs 57.83
  mS/cm at 20C, 0.5 M KCl) — needs Steve's answer; see the 2026-09-15 entry in
  `coordination/open-questions.md`.
