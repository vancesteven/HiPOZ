#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to prepend timestamps to Gamry data files based on their contents.

This script:
1. Reads each Gamry .txt file in the given directory
2. Extracts the timestamp from the second line
3. Converts it to military time (HHMMSS)
4. Renames the file to prepend this timestamp if it's not already there

Usage:
    python add_timestamps_to_filenames.py [data_directory]

Example:
    python add_timestamps_to_filenames.py data/20250815Cortes/ConductivityData_Default
"""

import os
import sys
import re
import shutil
from glob import glob
from datetime import datetime

def extract_timestamp_from_file(file_path):
    """
    Extract timestamp from the second line of a Gamry data file.

    Args:
        file_path (str): Path to Gamry data file

    Returns:
        str: Timestamp in HHMMSS format, or None if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip first line
            f.readline()
            # Read timestamp line (second line)
            timestamp_line = f.readline().strip()

            # Extract timestamp using regex - matches formats like "11:01:53.145 AM 2025-08-13"
            match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?\s+(AM|PM)\s+', timestamp_line)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                second = int(match.group(3))
                am_pm = match.group(4)

                # Convert to 24-hour format
                if am_pm.upper() == 'PM' and hour < 12:
                    hour += 12
                elif am_pm.upper() == 'AM' and hour == 12:
                    hour = 0

                # Format as HHMMSS
                return f"{hour:02d}{minute:02d}{second:02d}"

            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def add_timestamps_to_files(directory, dry_run=False):
    """
    Add timestamps to Gamry data files in directory.

    Args:
        directory (str): Path to directory containing Gamry data files
        dry_run (bool): If True, just print what would be done without actually renaming files

    Returns:
        tuple: (renamed_count, error_count, skipped_count)
    """
    # Ensure directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return 0, 1, 0

    # Find all .txt files in the directory (non-recursively)
    txt_files = glob(os.path.join(directory, "*.txt"))

    # Counters
    renamed_count = 0
    error_count = 0
    skipped_count = 0

    # Process each file
    for file_path in txt_files:
        dir_path, filename = os.path.split(file_path)

        # Skip if filename already starts with a timestamp pattern (HHMMSS_)
        if re.match(r'^\d{6}_', filename):
            print(f"Skipping (already has timestamp): {filename}")
            skipped_count += 1
            continue

        # Extract timestamp from file
        timestamp = extract_timestamp_from_file(file_path)

        if timestamp:
            # Create new filename with timestamp
            new_filename = f"{timestamp}_{filename}"
            new_path = os.path.join(dir_path, new_filename)

            # Rename the file
            action = "Would rename" if dry_run else "Renaming"
            print(f"{action}: {filename} -> {new_filename}")

            if not dry_run:
                try:
                    shutil.move(file_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {file_path}: {str(e)}")
                    error_count += 1
        else:
            print(f"Error: Could not extract timestamp from {filename}")
            error_count += 1

    return renamed_count, error_count, skipped_count

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python add_timestamps_to_filenames.py [data_directory]")
        print("Example: python add_timestamps_to_filenames.py data/20250815Cortes/ConductivityData_Default")
        return

    directory = sys.argv[1]

    # Confirm with user
    print(f"This will add timestamps to Gamry data files in directory:")
    print(f"  {directory}")
    print("Files already having timestamps (HHMMSS_) will be skipped.")
    confirm = input("Proceed? (y/n): ")

    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    # Process the files with dry run first
    print("\nAnalyzing files (dry run)...")
    renamed, errors, skipped = add_timestamps_to_files(directory, dry_run=True)

    if renamed == 0:
        print("\nNo files need renaming. Exiting.")
        return

    # Ask for confirmation to proceed with actual renaming
    print(f"\nFound {renamed} files to rename, {errors} with errors, {skipped} already with timestamps.")
    confirm = input("Proceed with renaming? (y/n): ")

    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    # Actually rename the files
    print("\nRenaming files...")
    renamed, errors, skipped = add_timestamps_to_files(directory, dry_run=False)

    print(f"\nDone! {renamed} files renamed, {errors} errors, {skipped} skipped.")

if __name__ == "__main__":
    main()