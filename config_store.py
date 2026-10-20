"""
Lossless read/merge/write of zAnalysis config files (CSV + JSON siblings).

AnalysisConfig.calibration_groups is an analysis view of a config: it drops
excluded rows, measurement conductivities and columns it does not use. Code
that writes a config back must not serialize that view, or every run erases
whatever the view left out. Writers go through this module instead:

    update_config(path, {filename: {field: value, ...}})

reads every row and column of the config, changes only the named fields of
the named rows (appending rows for new files), and writes the CSV and JSON
together. Rows, groups, exclusions, notes and extra columns (e.g. Z_Ohm from
imported workbooks) that the caller does not name are kept as they were.
"""
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from harmonize_config import FIELDNAMES, json_to_csv, rows_to_json, sync_rows

GROUP_KEYS = ('calibration_groups', 'calibrations')


def _siblings(path) -> Tuple[Path, Path]:
    path = Path(path)
    return path.with_suffix('.csv'), path.with_suffix('.json')


def read_rows(path) -> Tuple[List[str], List[dict]]:
    """All rows of a config as CSV-style dicts, plus the column order.
    Reads `path` itself (CSV or JSON); a missing file gives no rows."""
    path = Path(path)
    if not path.exists():
        return list(FIELDNAMES), []
    if path.suffix.lower() == '.json':
        rows = json_to_csv(path)
    else:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
            fieldnames = list(reader.fieldnames or [])
        return _fieldnames(fieldnames, rows), rows
    return _fieldnames([], rows), rows


def _fieldnames(start, rows) -> List[str]:
    names = list(start) or list(FIELDNAMES)
    for name in FIELDNAMES:
        if name not in names:
            names.append(name)
    for row in rows:
        names += [k for k in row if k and k not in names]
    return names


def _atomic_write(path: Path, write: Callable):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_rows(path, rows: List[dict], fieldnames: Optional[List[str]] = None,
               json_extra: Optional[dict] = None):
    """Write rows to both the CSV and JSON siblings of `path`.

    The JSON is written first, so the CSV is the newer file and the
    auto-harmonize step at load syncs CSV -> JSON (lossless). Top-level JSON
    keys other than the groups (descriptions, notes) are kept."""
    csv_path, json_path = _siblings(path)
    fieldnames = _fieldnames(fieldnames or [], rows)

    data, json_rows = {}, rows
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = {k: v for k, v in json.load(f).items() if k not in GROUP_KEYS}
            # keep entry keys only the JSON has; rows follow `rows`
            json_rows = sync_rows(rows, json_to_csv(json_path), keep_rows=False)
        except (OSError, ValueError, AttributeError):
            data = {}
    data.update(json_extra or {})
    data['calibration_groups'] = rows_to_json(json_rows)['calibration_groups']
    _atomic_write(json_path, lambda f: json.dump(data, f, indent=2))

    def write_csv(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None else v) for k, v in row.items()})
    _atomic_write(csv_path, write_csv)
    return csv_path, json_path


def merge_rows(rows: List[dict], updates: Dict[str, dict],
               default_group: Optional[str] = None,
               add_missing: bool = True) -> Tuple[int, int]:
    """Apply {filename: {field: value}} to rows in place.

    Only the named fields change. A filename with no row gets a new row in
    `default_group` (else the first group, else 'Group 1') when add_missing.
    Returns (rows updated, rows added)."""
    index = {r.get('filename'): r for r in rows}
    group = default_group or (rows[0].get('group_name') if rows else None) or 'Group 1'
    updated = added = 0
    for fname, fields in updates.items():
        row = index.get(fname)
        if row is None:
            if not add_missing:
                continue
            row = {'group_name': group, 'filename': fname, 'type': 'measurement'}
            rows.append(row)
            index[fname] = row
            added += 1
        else:
            updated += 1
        for k, v in fields.items():
            row[k] = '' if v is None else v
    return updated, added


def update_config(path, updates: Dict[str, dict], **kwargs) -> Tuple[int, int]:
    """Read the config at `path`, merge `updates`, write CSV + JSON.
    Returns (rows updated, rows added)."""
    fieldnames, rows = read_rows(path)
    counts = merge_rows(rows, updates, **kwargs)
    write_rows(path, rows, fieldnames)
    return counts
