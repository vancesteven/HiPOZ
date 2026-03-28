#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gamry timestamp utilities for HiPOZ - handles extracting and adding timestamps to filenames.

This module provides functions for extracting timestamps from Gamry data files and
renaming files to include these timestamps in their filenames.
"""

import os
import re
import shutil
from glob import glob
from datetime import datetime
import logging

# Set up logging
log = logging.getLogger('HiPOZ')

def extract_timestamp_from_file(file_path):
    """
    Extract timestamp from the second line of a Gamry data file.

    Args:
        file_path (str): Path to Gamry data file

    Returns:
        tuple: (timestamp_str, datetime_obj) or (None, None) if not found
               timestamp_str is in HHMMSS format
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip first line
            f.readline()
            # Read timestamp line (second line)
            timestamp_line = f.readline().strip()

            # Extract timestamp using regex - matches formats like "11:01:53.145 AM 2025-08-13"
            match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?\s+(AM|PM)\s+(\d{4}-\d{2}-\d{2})', timestamp_line)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                second = int(match.group(3))
                am_pm = match.group(4)
                date_str = match.group(5)

                # Convert to 24-hour format
                if am_pm.upper() == 'PM' and hour < 12:
                    hour += 12
                elif am_pm.upper() == 'AM' and hour == 12:
                    hour = 0

                # Format as HHMMSS
                timestamp_str = f"{hour:02d}{minute:02d}{second:02d}"

                # Parse the full datetime
                try:
                    date_time_obj = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:{second:02d}",
                                                      "%Y-%m-%d %H:%M:%S")
                    return timestamp_str, date_time_obj
                except ValueError:
                    log.warning(f"Could not parse full datetime in {file_path}")
                    return timestamp_str, None

            log.warning(f"Could not find timestamp in expected format in {file_path}")
            return None, None
    except Exception as e:
        log.error(f"Error reading file {file_path}: {str(e)}")
        return None, None

def add_timestamps_to_files(directory, recursive=False, dry_run=False):
    """
    Add timestamps to Gamry data files in directory.

    Args:
        directory (str): Path to directory containing Gamry data files
        recursive (bool): If True, process subdirectories recursively
        dry_run (bool): If True, just print what would be done without actually renaming files

    Returns:
        dict: Statistics containing counts of renamed, error, and skipped files
    """
    # Ensure directory exists
    if not os.path.isdir(directory):
        log.error(f"Directory '{directory}' does not exist")
        return {"error": "Directory not found"}

    # Find all .txt files
    pattern = os.path.join(directory, "**/*.txt") if recursive else os.path.join(directory, "*.txt")
    txt_files = glob(pattern, recursive=recursive)

    # Statistics
    stats = {
        "renamed": [],
        "errors": [],
        "skipped": [],
        "renamed_count": 0,
        "error_count": 0,
        "skipped_count": 0
    }

    # Process each file
    for file_path in txt_files:
        dir_path, filename = os.path.split(file_path)

        # Skip if filename already starts with a timestamp pattern (HHMMSS_)
        if re.match(r'^\d{6}_', filename):
            log.info(f"Skipping (already has timestamp): {filename}")
            stats["skipped"].append(file_path)
            stats["skipped_count"] += 1
            continue

        # Extract timestamp from file
        timestamp, _ = extract_timestamp_from_file(file_path)

        if timestamp:
            # Create new filename with timestamp
            new_filename = f"{timestamp}_{filename}"
            new_path = os.path.join(dir_path, new_filename)

            # Rename the file
            action = "Would rename" if dry_run else "Renaming"
            log.info(f"{action}: {filename} -> {new_filename}")

            if not dry_run:
                try:
                    shutil.move(file_path, new_path)
                    stats["renamed"].append((file_path, new_path))
                    stats["renamed_count"] += 1
                except Exception as e:
                    log.error(f"Error renaming {file_path}: {str(e)}")
                    stats["errors"].append((file_path, str(e)))
                    stats["error_count"] += 1
        else:
            log.error(f"Could not extract timestamp from {filename}")
            stats["errors"].append((file_path, "Could not extract timestamp"))
            stats["error_count"] += 1

    return stats

def process_directory_structure(root_dir, recursive=True, dry_run=False):
    """
    Process a directory structure looking for ConductivityData_Default directories.

    Args:
        root_dir (str): Root directory to start search
        recursive (bool): If True, search subdirectories recursively
        dry_run (bool): If True, just print what would be done without actually renaming files

    Returns:
        dict: Overall statistics with counts per directory
    """
    # Find ConductivityData_Default directories
    pattern = os.path.join(root_dir, "**/ConductivityData_Default")
    cond_dirs = glob(pattern, recursive=recursive)

    # Also include the root_dir if specified directly
    if os.path.basename(root_dir) == "ConductivityData_Default":
        cond_dirs.append(root_dir)

    # Deduplicate and sort
    cond_dirs = sorted(set(cond_dirs))

    # Overall statistics
    overall_stats = {
        "total_renamed": 0,
        "total_errors": 0,
        "total_skipped": 0,
        "directories_processed": 0,
        "per_directory": {}
    }

    # Process each ConductivityData_Default directory
    for cond_dir in cond_dirs:
        log.info(f"Processing directory: {cond_dir}")
        stats = add_timestamps_to_files(cond_dir, recursive=False, dry_run=dry_run)

        # Add to overall statistics
        overall_stats["total_renamed"] += stats["renamed_count"]
        overall_stats["total_errors"] += stats["error_count"]
        overall_stats["total_skipped"] += stats["skipped_count"]
        overall_stats["directories_processed"] += 1
        overall_stats["per_directory"][cond_dir] = {
            "renamed": stats["renamed_count"],
            "errors": stats["error_count"],
            "skipped": stats["skipped_count"]
        }

    return overall_stats

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Add timestamps to Gamry data files")
    parser.add_argument("directory", help="Directory containing Gamry data files")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Process subdirectories recursively")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--auto-detect", "-a", action="store_true",
                       help="Auto-detect ConductivityData_Default directories")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show verbose output")

    args = parser.parse_args()

    # Set up basic logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    if args.auto_detect:
        print(f"Auto-detecting ConductivityData_Default directories in {args.directory}...")
        stats = process_directory_structure(args.directory, recursive=args.recursive, dry_run=args.dry_run)

        print("\nOverall Statistics:")
        print(f"- Directories processed: {stats['directories_processed']}")
        print(f"- Total files renamed: {stats['total_renamed']}")
        print(f"- Total errors: {stats['total_errors']}")
        print(f"- Total files skipped (already had timestamps): {stats['total_skipped']}")

        if args.verbose:
            print("\nPer-Directory Statistics:")
            for dir_path, dir_stats in stats["per_directory"].items():
                print(f"  {dir_path}:")
                print(f"    - Renamed: {dir_stats['renamed']}")
                print(f"    - Errors: {dir_stats['errors']}")
                print(f"    - Skipped: {dir_stats['skipped']}")
    else:
        print(f"Processing files in {args.directory}...")
        stats = add_timestamps_to_files(args.directory, recursive=args.recursive, dry_run=args.dry_run)

        print("\nStatistics:")
        print(f"- Files renamed: {stats['renamed_count']}")
        print(f"- Errors: {stats['error_count']}")
        print(f"- Files skipped (already had timestamps): {stats['skipped_count']}")

    if args.dry_run:
        print("\nThis was a dry run. No files were actually renamed.")