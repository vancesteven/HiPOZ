# Audit log — hipozgenai

Append-only. One entry per completed action, newest at the bottom. Never edit
or delete existing entries. Format is defined in `AGENTS.md`.

## 2026-09-14 — claude-code — merge origin/main data into hipozgenai

- Merged `origin/main` into `hipozgenai` (merge commit `eab7098`).
- Brought in 665 new files, all lab-data text files (July 26 – Sept 11, 2026:
  `data/<date>/` Gamry sweeps, PressTemps, calibration logs). No Python code
  changed by the merge — main's only code edits were the matplotlib
  `get_cmap` fix, already present here from `16a66ed`, so both sides
  auto-merged identically. PyQt6 migration on this branch is intact.
- Note: main carries five data folders at the repo root (`20260801/` –
  `20260805/`) instead of under `data/`; inherited as-is, not moved.
- Status: verified (merge commit inspected; `git diff fa82535 HEAD
  --name-status -- '*.py'` empty; PyQt6 + colormaps imports confirmed in
  `gamry_HiPOZ.py`, `gamryTools.py`, `Pan_HiP.py`, `study_plots.py`).

## 2026-09-14 — claude-code — relocate root-level data folders into data/

- Moved `20260801/`–`20260805/` (PressTemps + calibration logs, inherited from
  main's upload) from the repo root to `data/<date>/`, matching every other
  dated data folder. `git mv` only; file contents unchanged.
- Same fix to be applied on `main`, per Steve's request.
- Status: verified (git status shows pure renames; no content diff).

## 2026-09-14 — claude-code — apply same folder fix on main and push both

- Applied the identical `git mv` on `main` (commit `5f4f723`, via the main
  checkout at ~/src/HiPOZ; its unrelated local changes left untouched) and
  pushed `main` to origin. Also pushed `hipozgenai` (`fa82535..4ad406a`).
- Both branches now hold identical renames, so future merges resolve cleanly.
- Status: verified (pushes confirmed by origin refs; rename-only commits).

## 2026-09-15 — claude-code — open PR #2 to propagate hipozgenai to main

- Steve asked to propagate this branch's functionality to `main`. Verified
  `hipozgenai` (`81b29a8`) and `main` (`5f4f723`) are both in sync with origin
  and that the merge is conflict-free (`git merge-tree --write-tree main
  hipozgenai` exits 0; the intentionally duplicated data-folder renames
  `4ad406a`/`5f4f723` auto-resolve).
- Opened PR #2 (hipozgenai → main): https://github.com/vancesteven/HiPOZ/pull/2
- `gh pr merge 2 --merge` was denied by the session permission gate
  ("merge without review"), so the merge is left for Steve.
- Uncommitted working-tree changes (modified cortes2026 plot PDFs and
  `tests/test_gui_reorganization.py`; ~30 untracked files incl. root-level
  planning .md notes, `JesusCortes/`, Aug 2025 Cortes data, mahboub2026 drafts)
  are NOT part of the PR — they were never committed to the branch. Needs a
  decision on what to keep; logged in `coordination/open-questions.md`.
- Status: implemented, unverified (PR open and mergeable; propagation not yet
  observed on `origin/main`).

## 2026-09-15 — claude-code — PR #2 merged; hipozgenai propagated to main

- Steve merged PR #2 (`gh pr merge 2 --merge`; the session permission gate had
  blocked claude-code from merging it directly). Merge commit `351853f` on
  `origin/main`.
- Status: verified (`git merge-base --is-ancestor fd3718d origin/main` passes,
  so main contains the full hipozgenai branch, bookkeeping commit included).
- The local main checkout at ~/src/HiPOZ is now behind origin/main and needs a
  `git pull` there; not touched from this worktree.

## 2026-09-15 — claude-code — curate uncommitted working tree per Steve

- Steve chose per-group dispositions for the ~30 uncommitted files left out of
  PR #2. Executed: (1) `978540a` — .gitignore policy, PyQt6 fix to
  test_gui_reorganization.py, CLAUDE.md/AGENTS.md/CODEX-QUEUE/open-questions
  now tracked, four new test suites with mechanical plumbing repairs, two
  plotting docs moved under docs/, plot_all.sh + test_mccleskey_cortes.sh;
  (2) `e7b5f6f` — Aug 18-20 2025 Cortes data folders, JesusCortes/ (minus
  Office lock file), pre-20250624 benchtop CSV snapshot; (3) `af53f12` —
  mahboub2026 revision .tex drafts + generate_correct_csv.py.
- Discarded: regenerated cortes_plots PDFs (byte-identical sizes; restored
  from HEAD); deleted six root session-note .md files. Left untracked per
  Steve: Mahboub2026BenchtopData_INCORRECT.csv, mahboub2026_plots.py.save,
  precision_formatting_report.md.
- Tests run in a scratch venv (Linux container, no TeX, no libEGL):
  test_benchtop_data 5/6 (KCl reference mismatch logged in open-questions,
  NOT adjusted), test_latex_tables all pass after fixing a literal-string
  assertion, test_plot_generation 5/7 (two need LaTeX), GUI suites not
  runnable here. Status: verified for the curation itself; the test suites
  are implemented, unverified pending a macOS run.
