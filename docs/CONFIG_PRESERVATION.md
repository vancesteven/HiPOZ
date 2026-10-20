# Config preservation: analysis never erases zAnalysis contents

Updated: 2026-10-20

A zAnalysis config is a CSV and a JSON with the same name in
`data/<folder>/`. People edit them (the CSV in Excel), import them from
workbooks, or annotate them (exclusions with reasons, notes). Running an
analysis must change only the values the analysis computes.

## The bug (before 2026-10-20)

Each writer below rebuilt the files from `AnalysisConfig.calibration_groups`
or from the GUI table, not from the file. Those views hold only what the
analysis uses, so every write erased the rest.

| Writer | What it erased |
|---|---|
| Headless write-back (`headless_analysis._update_config_with_conductivities`) | excluded rows and their reasons, extra columns |
| GUI save (`DataSelector.save_gui_state_to_config`, after every edit) | every `exclude` flag, groups (collapsed to one), rows without data, extra columns; the JSON was rewritten under the old `calibrations` key |
| `--cal-map` (`calibration_sharing.write_config`) | excluded rows, extra columns |
| Load-time sync (`harmonize_config(auto_detect=True)`), JSON → CSV whenever the JSON was newer | extra columns, measurement σ (which the GUI uses to restore associations), anything a stale JSON lacked |

## The rule now

All writers go through `config_store.py`:

- `update_config(path, {filename: {field: value}})` reads every row and
  column of the file, changes only the named fields of the named rows
  (appending rows for new files), and writes the CSV and JSON together.
- `write_rows` writes the JSON first, then the CSV, both atomically. The CSV
  is therefore the newer file, and the next load syncs CSV → JSON. Top-level
  JSON keys and entry keys that only the JSON has are kept.

Writers:
- **Headless:** sets only `conductivity_Sm` on measured rows. The
  `_analyzed` backups are full copies.
- **GUI:** merges only the fields its table owns (P, T, type, Z, Z±, σ, σ±,
  comp, w_ppt, w_molal). A chiller-bath row (P_MPa = 0) keeps its P/T
  unless the GUI also shows 0 MPa.
- **`--cal-map`:** removes and adds only the rows tagged `borrowed_from:`.

Load-time sync (`harmonize_config.sync_rows`):
- **JSON → CSV is additive.** Rows and columns only the CSV has are kept,
  and a blank JSON value does not blank a filled CSV value. A stale or
  older-format JSON therefore cannot delete CSV data.
- **CSV → JSON** lets the CSV win for every column it has. Rows deleted from
  the CSV stay deleted.

Trade-off: clearing a value in the JSON alone does not propagate to the CSV.
Edit the CSV to clear values.

## Verification

- `tests/test_config_store.py`: headless with CSV and JSON inputs, GUI save,
  `--cal-map` re-association, the JSON round trip, and a stale JSON newer
  than the CSV.
- `tests/test_curate_diaz_2026.py`
- Real data: a headless run on 20260923, and on 20260925 with
  `--cal-map 20260925=20260924` starting from a stale JSON. Rows, columns,
  exclusions and notes were unchanged; the only changes were σ (within
  0.044% of the previous values) and the borrowed standard row.
- The GUI save is tested by calling the method directly. It has not been
  exercised in the running GUI (`implemented, unverified` there).
