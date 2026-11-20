# Config preservation: analysis never erases zAnalysis contents

Updated: 2026-10-20

A zAnalysis config is a CSV and a JSON with the same name in
`data/<folder>/`. People edit them (the CSV in Excel), import them from
workbooks, or annotate them (exclusions with reasons, notes). Running an
analysis must change only the values the analysis computes.

The CSV is the source of truth. The JSON is a mirror of the CSV, regenerated
from it. Edit the CSV (e.g. in Excel), not the JSON. Edits made only in the
JSON are overwritten at the next load.

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
- `write_rows` writes the CSV, then rewrites the JSON as its exact mirror,
  both atomically. Top-level JSON keys other than the groups (e.g.
  `description`) are kept.
- `read_rows` reads the CSV sibling whenever it exists, else the JSON.

Writers:
- **Headless:** sets only `conductivity_Sm` on measured rows. The
  `_analyzed` backups are full copies.
- **GUI:** merges only the fields its table owns (P, T, type, Z, Z±, σ, σ±,
  comp, w_ppt, w_molal). A chiller-bath row (P_MPa = 0) keeps its P/T
  unless the GUI also shows 0 MPa.
- **`--cal-map`:** removes and adds only the rows tagged `borrowed_from:`.

Load-time sync (`harmonize_config(auto_detect=True)`):
- **If the CSV exists, it always wins.** The CSV is synced to the JSON
  regardless of which file is newer. The JSON is rewritten as an exact mirror
  of the CSV rows. Entry keys that only the JSON had are not kept.
- **JSON → CSV at load happens only for a folder with no CSV yet.**
- **Manual JSON → CSV harmonize into an existing CSV**
  (`harmonize_config.sync_rows`) is additive. Rows and columns only the CSV
  has are kept, and a blank JSON value does not blank a filled CSV value.
- `AnalysisConfig` given a JSON path whose CSV sibling exists loads the CSV
  and sets `config_path` to the CSV.

Trade-off: edits made only in the JSON are lost at the next load, and the
manual JSON → CSV harmonize cannot clear a CSV value. Edit the CSV to change
or clear values.

## Verification

- `tests/test_config_store.py`: headless with CSV and JSON inputs, GUI save,
  `--cal-map` re-association, the JSON round trip, and a stale JSON newer
  than the CSV. Also `test_csv_is_source_of_truth_at_load`,
  `test_explicit_json_to_csv_is_additive`, and `test_json_only_folder_still_converts`.
- `tests/test_curate_diaz_2026.py`
- Full suite: 116 pass, 1 known unrelated failure (`test_kcl_extraction`).
- Real data: a deliberately corrupted JSON (one row, old `calibrations` key)
  made newer than `data/20260923/zAnalysis20260923.csv`. A headless run loaded
  the CSV and left it byte-identical, and rewrote the JSON as a 49-row mirror
  with 3 excluded entries.
- Real data: a headless run on 20260923, and on 20260925 with
  `--cal-map 20260925=20260924` starting from a stale JSON. Rows, columns,
  exclusions and notes were unchanged; the only changes were σ (within
  0.044% of the previous values) and the borrowed standard row.
- The GUI save is tested by calling the method directly. It has not been
  exercised in the running GUI (`implemented, unverified` there).
