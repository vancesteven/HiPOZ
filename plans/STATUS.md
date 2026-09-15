# Status — hipozgenai

Updated: 2026-09-15

Refresh the `Updated:` line and the affected sections in any session that
pushes commits, integrates artifacts, or changes a queue.

## Current focus

Working-tree curation after the PR #2 propagation; second propagation pass
(PR #3) pending Steve's merge.

## In flight

| Item | Owner | Status | Artifact |
|---|---|---|---|
| Merge origin/main lab data into hipozgenai | claude-code | verified | merge commit `eab7098`; audit entry 2026-09-14 in `coordination/audit.md` |
| Propagate hipozgenai to main | claude-code | verified | PR #2 merged by Steve 2026-09-15; merge commit `351853f` on `origin/main` contains branch tip `fd3718d` (`git merge-base --is-ancestor` confirmed) |
| Curate uncommitted working tree per Steve's decisions | claude-code | verified | commits `978540a` (tests/infra/docs), `e7b5f6f` (Aug 2025 Cortes + JesusCortes data), `af53f12` (mahboub2026 drafts); audit entry 2026-09-15 |
| New test suites pass in full project env (TeX + GUI) | claude-code | implemented, unverified | 20/23 collected tests pass in the Linux container; `test_plot_export`, `test_multi_panel_plot` need LaTeX and the two GUI suites need libEGL — run `pytest tests/` on macOS |

Status must be one of `verified` / `implemented, unverified` / `not implemented`.

## Blockers

- `test_kcl_extraction` fails on a reference-value question (55.88 vs 57.83
  mS/cm at 20C, 0.5 M KCl) — needs Steve's answer; see the 2026-09-15 entry in
  `coordination/open-questions.md`.
