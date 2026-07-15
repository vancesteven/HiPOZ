"""
Calibration sharing for HiPOZ multi-folder analysis.

Some measurement days do not include their own conductivity standards.
This module lets you explicitly borrow calibration standards from another
day's data folder so that every folder's zAnalysis config becomes
self-contained.

Design decisions (see project memory):
  1. Borrowed standard .txt files are PHYSICALLY copied into the target
     folder (retaining their original filename) so the folder can be
     re-processed on its own.
  2. There is NO auto-detection of standards and NO assumed association.
     The user must state associations explicitly via --cal-map TARGET=SOURCE.
  3. The association is undoable / re-assignable: borrowed entries are
     tagged in the config ("borrowed_from:<source>") and both the config
     entries and the physically copied files are removed before a new
     association is applied.
"""

import os
import csv
import json
import shutil
import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger('HiPOZ')

# Prefix stored in a standard entry's "notes" field to mark it as borrowed.
BORROWED_TAG = 'borrowed_from:'

# CSV column order shared with headless_analysis / analysis_config.
CONFIG_FIELDNAMES = ['group_name', 'filename', 'P_MPa', 'T_K', 'type',
                     'conductivity_Sm', 'comp', 'w_ppt', 'w_molal',
                     'exclude', 'notes']


def parse_cal_map(cal_map_args: Optional[List[str]]) -> Dict[str, str]:
    """
    Parse --cal-map arguments of the form TARGET=SOURCE.

    Args:
        cal_map_args: List like ['20250819Cortes=20250818Cortes', ...]

    Returns:
        Dict {target_folder: source_folder}
    """
    mapping: Dict[str, str] = {}
    if not cal_map_args:
        return mapping

    for item in cal_map_args:
        if '=' not in item:
            log.error(f"Invalid --cal-map entry '{item}'. Expected TARGET=SOURCE.")
            continue
        target, source = item.split('=', 1)
        target = target.strip()
        source = source.strip()
        if not target or not source:
            log.error(f"Invalid --cal-map entry '{item}'. Expected TARGET=SOURCE.")
            continue
        if target == source:
            log.warning(f"Skipping cal-map '{item}': target and source are identical.")
            continue
        mapping[target] = source

    return mapping


def _entry_filename(entry) -> str:
    """Return the filename of a standard/measurement entry (string or dict)."""
    if isinstance(entry, dict):
        return entry.get('filename', entry.get('file', ''))
    return str(entry)


def _entry_notes(entry) -> str:
    """Return the notes of an entry (only dict entries carry notes)."""
    if isinstance(entry, dict):
        return entry.get('notes', '') or ''
    return ''


def _is_borrowed(entry) -> bool:
    return _entry_notes(entry).startswith(BORROWED_TAG)


def _find_data_file(folder: str, basename: str) -> Optional[str]:
    """
    Locate a data file by basename within a data folder's Conductivity*
    subdirectories.

    Args:
        folder: Folder name under data/ (e.g. '20250818Cortes')
        basename: File name to locate

    Returns:
        Full path if found, else None.
    """
    matches = glob(os.path.join('data', folder, 'Conductivity*', basename))
    return matches[0] if matches else None


def _collect_source_standards(source_config) -> List[dict]:
    """
    Collect all non-excluded standard entries (as dicts) from a source config.

    Only standards that carry a conductivity value are useful for calibration,
    but we copy metadata verbatim and let the analysis step validate.
    """
    standards: List[dict] = []
    seen = set()

    for group in source_config.calibration_groups:
        for std in group.get('standards', []):
            fname = _entry_filename(std)
            if not fname or fname in seen:
                continue
            # Skip standards that were themselves borrowed into the source.
            if _is_borrowed(std):
                continue
            seen.add(fname)

            if isinstance(std, dict):
                entry = dict(std)
            else:
                entry = {'filename': fname}
            standards.append(entry)

    return standards


def _target_conductivity_dir(target: str) -> str:
    """
    Choose (and create) a Conductivity* subdirectory in the target folder to
    receive copied standard files. Prefers an existing ConductivityData_Default.
    """
    default_dir = os.path.join('data', target, 'ConductivityData_Default')
    existing = sorted(glob(os.path.join('data', target, 'Conductivity*')))

    if os.path.isdir(default_dir):
        chosen = default_dir
    elif existing:
        chosen = existing[0]
    else:
        chosen = default_dir

    os.makedirs(chosen, exist_ok=True)
    return chosen


def _remove_previous_borrowed(target_config, target: str) -> int:
    """
    Remove previously borrowed standard entries and their physically copied
    files from the target folder. This enables clean re-association (undo).

    Returns:
        Number of borrowed entries removed.
    """
    removed = 0

    for group in target_config.calibration_groups:
        kept = []
        for std in group.get('standards', []):
            if _is_borrowed(std):
                fname = _entry_filename(std)
                path = _find_data_file(target, fname)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        log.info(f"  Removed previously borrowed file: {fname}")
                    except OSError as e:
                        log.warning(f"  Could not remove {path}: {e}")
                removed += 1
            else:
                kept.append(std)
        group['standards'] = kept

    return removed


def write_config(config_path: str, calibration_groups: List[dict]):
    """
    Write calibration_groups to both CSV and JSON, keyed off the CSV path.

    Args:
        config_path: Path to the CSV config file. A sibling .json is written too.
    """
    csv_path = Path(config_path)
    if csv_path.suffix.lower() != '.csv':
        csv_path = csv_path.with_suffix('.csv')
    json_path = csv_path.with_suffix('.json')

    # --- CSV ---
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CONFIG_FIELDNAMES)
        writer.writeheader()

        for group in calibration_groups:
            group_name = group.get('name', 'Group 1')
            for kind in ('standards', 'measurements'):
                type_label = 'standard' if kind == 'standards' else 'measurement'
                for entry in group.get(kind, []):
                    if isinstance(entry, dict):
                        e = entry
                    else:
                        e = {'filename': str(entry)}
                    writer.writerow({
                        'group_name': group_name,
                        'filename': e.get('filename', ''),
                        'P_MPa': e.get('P_MPa', ''),
                        'T_K': e.get('T_K', ''),
                        'type': type_label,
                        'conductivity_Sm': e.get('conductivity_Sm', ''),
                        'comp': e.get('comp', ''),
                        'w_ppt': e.get('w_ppt', ''),
                        'w_molal': e.get('w_molal', ''),
                        'exclude': 'x' if e.get('exclude') else '',
                        'notes': e.get('notes', ''),
                    })

    # --- JSON ---
    with open(json_path, 'w') as f:
        json.dump({
            'calibration_groups': calibration_groups,
            'generated_by': 'HiPOZ calibration_sharing',
        }, f, indent=2)

    log.info(f"  Wrote config: {csv_path.name} (+ .json)")


def borrow_calibration(target: str, source: str, target_config, source_config,
                       target_config_path: str) -> bool:
    """
    Copy calibration standards from a source folder into a target folder.

    Physically copies the standard .txt files into the target folder and adds
    matching standard entries (tagged as borrowed) to the target config, after
    removing any previously borrowed standards (clean re-association).

    Args:
        target: Target folder name under data/ (borrows calibration)
        source: Source folder name under data/ (provides calibration)
        target_config: AnalysisConfig for the target folder (modified in place)
        source_config: AnalysisConfig for the source folder
        target_config_path: Path to the target's CSV config (json sibling written too)

    Returns:
        True if calibration was borrowed successfully, False otherwise.
    """
    log.info(f"Borrowing calibration: {target} <- {source}")

    source_standards = _collect_source_standards(source_config)
    if not source_standards:
        log.error(f"  Source '{source}' has no standards to borrow.")
        return False

    # Clean out any previously borrowed calibration first (undo/replace).
    _remove_previous_borrowed(target_config, target)

    dest_dir = _target_conductivity_dir(target)

    copied = 0
    borrowed_entries: List[dict] = []
    for std in source_standards:
        fname = _entry_filename(std)
        src_path = _find_data_file(source, fname)
        if not src_path:
            log.warning(f"  Source standard file not found on disk, skipping: {fname}")
            continue

        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        try:
            shutil.copy2(src_path, dest_path)
            copied += 1
            log.info(f"  Copied standard: {fname}")
        except OSError as e:
            log.warning(f"  Failed to copy {fname}: {e}")
            continue

        entry = dict(std)
        entry['filename'] = os.path.basename(src_path)
        entry['notes'] = f"{BORROWED_TAG}{source}"
        borrowed_entries.append(entry)

    if not borrowed_entries:
        log.error(f"  No standard files could be copied from '{source}'.")
        return False

    # Attach borrowed standards to the target config. Ensure at least one group.
    if not target_config.calibration_groups:
        target_config.calibration_groups = [{
            'name': 'Group 1', 'standards': [], 'measurements': []
        }]

    for group in target_config.calibration_groups:
        group.setdefault('standards', [])
        for entry in borrowed_entries:
            group['standards'].append(dict(entry))

    write_config(target_config_path, target_config.calibration_groups)

    log.info(f"  Borrowed {copied} standard(s) from '{source}' into '{target}'")
    return True
