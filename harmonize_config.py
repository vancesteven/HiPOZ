#!/usr/bin/env python
"""
Harmonize calibration config files between CSV and JSON formats.

Usage:
    python harmonize_config.py data/20250815Mahboub2026/zAnalysis20250815Mahboub2026.csv
    python harmonize_config.py data/20250815Mahboub2026/zAnalysis20250815Mahboub2026.json

Reads the specified file and creates/updates the matching format.
"""

import sys
import json
import csv
from pathlib import Path
import argparse

FIELDNAMES = ['group_name', 'filename', 'P_MPa', 'T_K', 'type',
              'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes']


def _extras(entry):
    """Columns outside FIELDNAMES (e.g. Z_Ohm in imported workbooks), kept
    verbatim so a CSV -> JSON -> CSV round trip does not drop them."""
    return {k: v for k, v in entry.items()
            if k and k not in FIELDNAMES and v not in (None, '')}


def sync_rows(new_rows: list, old_rows: list, keep_rows: bool = True) -> list:
    """Rows of the file being synced from (new) merged onto the file being
    replaced (old). Columns only old has are kept.

    keep_rows=True (JSON -> CSV): additive. Rows only old has are kept, and a
    blank new value does not overwrite a filled old one, so a stale or
    older-format JSON cannot delete CSV data (rows, exclusions, notes).
    keep_rows=False (CSV -> JSON): the CSV, the file people edit, wins
    outright for every column it has, and rows deleted from it stay deleted.
    """
    old = {r.get('filename'): r for r in old_rows}
    seen = set()
    out = []
    for r in new_rows:
        merged = dict(old.get(r.get('filename'), {}))
        merged.update({k: v for k, v in r.items()
                       if not keep_rows or v not in ('', None) or k not in merged})
        out.append(merged)
        seen.add(r.get('filename'))
    kept = [r for r in old_rows if r.get('filename') not in seen] if keep_rows else []
    if kept:
        print(f"NOTE: kept {len(kept)} row(s) missing from the source: "
              + ', '.join(str(r.get('filename')) for r in kept[:5]))
    return out + kept


def _read_csv_rows(csv_path: Path) -> list:
    with open(csv_path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def csv_to_json(csv_path: Path) -> dict:
    """Convert CSV calibration config to JSON structure."""

    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return rows_to_json(rows)


def rows_to_json(rows: list) -> dict:
    """Convert CSV-style row dicts (values str or numbers) to JSON structure."""

    # Group by calibration group
    groups = {}
    for row in rows:
        row = {k: '' if v is None else str(v) for k, v in row.items() if k}
        group_name = row.get('group_name') or 'Group 1'
        if group_name not in groups:
            groups[group_name] = {
                'name': group_name,
                'standards': [],
                'measurements': []
            }

        # Build entry
        entry = {
            'filename': row.get('filename', '')
        }

        # Add optional fields if present
        if row.get('P_MPa'):
            try:
                entry['P_MPa'] = float(row['P_MPa']) if '.' in row['P_MPa'] else int(row['P_MPa'])
            except ValueError:
                entry['P_MPa'] = row['P_MPa']

        if row.get('T_K'):
            try:
                entry['T_K'] = float(row['T_K']) if '.' in row['T_K'] else int(row['T_K'])
            except ValueError:
                entry['T_K'] = row['T_K']

        if row.get('conductivity_Sm'):
            try:
                entry['conductivity_Sm'] = float(row['conductivity_Sm'])
            except ValueError:
                pass

        if row.get('comp'):
            entry['comp'] = row['comp']

        if row.get('w_ppt'):
            try:
                entry['w_ppt'] = float(row['w_ppt'])
            except ValueError:
                entry['w_ppt'] = row['w_ppt']

        if row.get('w_molal'):
            try:
                entry['w_molal'] = float(row['w_molal'])
            except ValueError:
                entry['w_molal'] = row['w_molal']

        if row.get('exclude') and row['exclude'].lower() in ['x', 'yes', 'true', '1']:
            entry['exclude'] = True

        if row.get('notes'):
            entry['notes'] = row['notes']

        entry.update(_extras(row))

        # Add to appropriate list
        file_type = row.get('type', 'measurement').lower()
        if file_type == 'standard':
            groups[group_name]['standards'].append(entry)
        else:
            groups[group_name]['measurements'].append(entry)

    # Build final JSON structure
    json_data = {
        'calibration_groups': list(groups.values())
    }

    return json_data


def json_to_csv(json_path: Path) -> list:
    """Convert JSON calibration config to CSV rows."""

    # Read JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract groups
    groups = data.get('calibration_groups', data.get('calibrations', []))

    rows = []
    for group in groups:
        group_name = group.get('name', 'Group 1')

        # Process standards
        for std in group.get('standards', []):
            if not isinstance(std, dict):
                std = {'filename': str(std)}
            row = {
                'group_name': group_name,
                'filename': std.get('filename', ''),
                'P_MPa': std.get('P_MPa', ''),
                'T_K': std.get('T_K', ''),
                'type': 'standard',
                'conductivity_Sm': std.get('conductivity_Sm', ''),
                'comp': std.get('comp', ''),
                'w_ppt': std.get('w_ppt', ''),
                'w_molal': std.get('w_molal', ''),
                'exclude': 'x' if std.get('exclude') else '',
                'notes': std.get('notes', '')
            }
            row.update(_extras(std))
            rows.append(row)

        # Process measurements
        for meas in group.get('measurements', []):
            if not isinstance(meas, dict):
                meas = {'filename': str(meas)}
            row = {
                'group_name': group_name,
                'filename': meas.get('filename', ''),
                'P_MPa': meas.get('P_MPa', ''),
                'T_K': meas.get('T_K', ''),
                'type': 'measurement',
                'conductivity_Sm': meas.get('conductivity_Sm', ''),  # computed; the GUI restores it
                'comp': meas.get('comp', ''),
                'w_ppt': meas.get('w_ppt', ''),
                'w_molal': meas.get('w_molal', ''),
                'exclude': 'x' if meas.get('exclude') else '',
                'notes': meas.get('notes', '')
            }
            row.update(_extras(meas))
            rows.append(row)

    return rows


def harmonize_config(file_path: Path, force: bool = False, verbose: bool = True, auto_detect: bool = False):
    """
    Harmonize config file - create matching format.

    Args:
        file_path: Path to CSV or JSON config file
        force: Overwrite existing matching file
        verbose: Print status messages
        auto_detect: If True, detect which file is newer and sync from that one

    Returns:
        Path to created/updated file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return None

    is_csv = file_path.suffix.lower() == '.csv'

    # Auto-detect: check which file is newer
    if auto_detect:
        counterpart_path = file_path.with_suffix('.json' if is_csv else '.csv')

        if counterpart_path.exists():
            file_mtime = file_path.stat().st_mtime
            counterpart_mtime = counterpart_path.stat().st_mtime

            if counterpart_mtime > file_mtime:
                # Counterpart is newer, switch to harmonize from that one
                if verbose:
                    from datetime import datetime
                    file_time = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    counter_time = datetime.fromtimestamp(counterpart_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Auto-detect: {counterpart_path.name} is newer ({counter_time} vs {file_time})")
                    print(f"Syncing from {counterpart_path.name} → {file_path.name}")

                file_path = counterpart_path
                is_csv = file_path.suffix.lower() == '.csv'
            elif verbose:
                from datetime import datetime
                file_time = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                counter_time = datetime.fromtimestamp(counterpart_mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Auto-detect: {file_path.name} is newer ({file_time} vs {counter_time})")
                print(f"Syncing from {file_path.name} → {counterpart_path.name}")

    if is_csv:
        # CSV → JSON
        if verbose:
            print(f"Reading CSV: {file_path}")

        json_path = file_path.with_suffix('.json')

        if json_path.exists() and not force:
            print(f"WARNING: JSON file already exists: {json_path}")
            print("Use --force to overwrite")
            return None

        rows = _read_csv_rows(file_path)
        json_data = {}
        if json_path.exists():
            try:
                with open(json_path) as f:
                    json_data = {k: v for k, v in json.load(f).items()
                                 if k not in ('calibration_groups', 'calibrations')}
                rows = sync_rows(rows, json_to_csv(json_path), keep_rows=False)
            except (OSError, ValueError, AttributeError) as e:
                print(f"WARNING: could not merge existing {json_path}: {e}")
        json_data['calibration_groups'] = rows_to_json(rows)['calibration_groups']

        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        if verbose:
            print(f"✓ Created JSON: {json_path}")
            print(f"  Groups: {len(json_data['calibration_groups'])}")
            for group in json_data['calibration_groups']:
                n_std = len(group['standards'])
                n_meas = len(group['measurements'])
                print(f"  {group['name']}: {n_std} standards, {n_meas} measurements")

        return json_path

    else:
        # JSON → CSV
        if verbose:
            print(f"Reading JSON: {file_path}")

        csv_rows = json_to_csv(file_path)

        # Determine output path
        csv_path = file_path.with_suffix('.csv')

        if csv_path.exists() and not force:
            print(f"WARNING: CSV file already exists: {csv_path}")
            print("Use --force to overwrite")
            return None

        fieldnames = list(FIELDNAMES)
        if csv_path.exists():
            with open(csv_path, newline='') as f:
                fieldnames = list(csv.DictReader(f).fieldnames or FIELDNAMES)
            fieldnames += [k for k in FIELDNAMES if k not in fieldnames]
            csv_rows = sync_rows(csv_rows, _read_csv_rows(csv_path))

        # Write CSV
        for row in csv_rows:
            fieldnames += [k for k in row if k and k not in fieldnames]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)

        if verbose:
            print(f"✓ Created CSV: {csv_path}")
            print(f"  Total rows: {len(csv_rows)}")

            # Count by group
            groups = {}
            for row in csv_rows:
                gname = row['group_name']
                if gname not in groups:
                    groups[gname] = {'standards': 0, 'measurements': 0}
                if row['type'] == 'standard':
                    groups[gname]['standards'] += 1
                else:
                    groups[gname]['measurements'] += 1

            for gname, counts in groups.items():
                print(f"  {gname}: {counts['standards']} standards, {counts['measurements']} measurements")

        return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Harmonize calibration config files between CSV and JSON formats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CSV to JSON:
  python harmonize_config.py data/20250815/zAnalysis20250815.csv

  # Convert JSON to CSV:
  python harmonize_config.py data/20250815/zAnalysis20250815.json

  # Overwrite existing file:
  python harmonize_config.py --force data/20250815/zAnalysis20250815.csv

  # Quiet mode (minimal output):
  python harmonize_config.py --quiet data/20250815/zAnalysis20250815.csv

  # Harmonize all configs in a directory:
  for f in data/*/zAnalysis*.csv; do python harmonize_config.py "$f"; done
"""
    )

    parser.add_argument(
        'file',
        help='Path to CSV or JSON config file to harmonize'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Overwrite existing matching file'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode - minimal output'
    )

    parser.add_argument(
        '-a', '--auto',
        action='store_true',
        help='Auto-detect which file is newer and sync from that one'
    )

    args = parser.parse_args()

    result = harmonize_config(
        Path(args.file),
        force=args.force,
        verbose=not args.quiet,
        auto_detect=args.auto
    )

    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
