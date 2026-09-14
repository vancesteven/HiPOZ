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
