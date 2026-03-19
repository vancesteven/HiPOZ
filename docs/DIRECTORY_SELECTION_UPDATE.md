# Directory Selection Update

## Overview

Updated HiPOZ to remove hardcoded directory names from Python code. Users can now specify directories via:
1. **GUI dialog** - Visual directory selection
2. **Command line** - `--dates` flag

Students no longer need to edit `gamry_HiPOZOZ.py` to analyze their data.

## Changes Made

### 1. Removed Hardcoded Dates

**Before:**
```python
dates = ['20250813Mahboub2026']  # Students had to edit this
```

**After:**
```python
dates = []  # Empty by default - prompts for selection
```

### 2. Added Directory Selection Dialog

New function `select_data_directories()`:
- Opens GUI file dialog
- Allows single or multiple directory selection
- Starts in `data/` directory
- Returns list of directory names

**Usage:**
```python
# Called automatically when no --dates specified
selected_dirs = select_data_directories()
```

### 3. Updated main() Function

**Flow:**
1. Parse command line arguments
2. Initialize QApplication (needed for dialogs)
3. Check if `--dates` specified
4. If not, check if `dates` list has entries
5. If empty, prompt user with directory selection dialog
6. If user cancels, exit gracefully
7. Proceed with selected directories

**Code:**
```python
def main():
    args = parse_arguments()
    app = QApplication(sys.argv)

    # Determine directories
    dates_to_process = args.dates if args.dates else dates

    # Prompt if empty
    if not dates_to_process:
        dates_to_process = select_data_directories()
        if not dates_to_process:
            log.error("No directories selected. Exiting.")
            sys.exit(0)

    log.info(f"Processing directories: {dates_to_process}")
    # ... continue with analysis
```

### 4. Enhanced Command Line Help

**Updated examples:**
```bash
# Run with GUI - will prompt for directory selection:
python gamry_HiPOZOZ.py

# Specify directory(ies) from command line:
python gamry_HiPOZOZ.py --dates 20250815Mahboub

# Process multiple directories:
python gamry_HiPOZOZ.py --dates 20250813 20250814 20250815

# Headless mode with specific directory:
python gamry_HiPOZOZ.py --headless --dates 20250815Mahboub

# Run with specific config file:
python gamry_HiPOZOZ.py --config data/20250815/zAnalysis20250815.csv
```

**Updated --dates help text:**
```
--dates: Specify data directory(ies) to process (e.g., --dates 20250815Mahboub).
         If not provided, GUI will prompt for directory selection.
```

## Usage Examples

### For Students

**Scenario 1: GUI with Directory Selection**
```bash
$ python gamry_HiPOZOZ.py
```
1. Program starts
2. Dialog opens: "Select Data Directory(ies)"
3. Navigate to `data/` folder
4. Click on directory (e.g., `20250815Mahboub2026`)
5. Click "Open" or "Select Folder"
6. Analysis begins

**Scenario 2: Command Line**
```bash
$ python gamry_HiPOZOZ.py --dates 20250815Mahboub2026
```
- No GUI dialog
- Directly processes specified directory
- Faster for experienced users

**Scenario 3: Multiple Directories**
```bash
$ python gamry_HiPOZOZ.py --dates 20250813 20250814 20250815
```
- Processes all three directories in sequence
- Useful for batch analysis

**Scenario 4: Headless Automation**
```bash
$ python gamry_HiPOZOZ.py --headless --dates 20250815Mahboub2026
```
- No GUI
- Requires config file with standards specified
- Fast automated analysis

### For Instructors

**Setting up for students:**
1. Tell students: `python gamry_HiPOZOZ.py`
2. They select their data folder
3. Done! No Python editing needed

**Batch processing:**
```bash
# Process all student directories
python gamry_HiPOZOZ.py --dates Student1_20250815 Student2_20250815 Student3_20250815
```

## Directory Selection Dialog

### Features
- **File browser interface** - Familiar to all users
- **Starts in data/ folder** - Convenient default location
- **Visual navigation** - Click through folders
- **Multiple selection** - Can select multiple directories (implementation may vary by platform)
- **Cancel option** - Exits gracefully if user cancels

### Platform Behavior
- **macOS:** Native macOS directory picker
- **Windows:** Native Windows folder browser
- **Linux:** Qt folder dialog

## Benefits

### For Students
✅ **No Python editing** - Don't need to understand code
✅ **Visual interface** - Click to select folder
✅ **Error prevention** - Can't accidentally break code
✅ **Clear workflow** - Run → Select → Analyze

### For Instructors
✅ **Easier teaching** - One command for all students
✅ **Less support** - No "I broke the code" issues
✅ **Reproducible** - Same command works every time
✅ **Scriptable** - Can still automate with --dates flag

### For Workflows
✅ **Flexible** - GUI or command line
✅ **Batch processing** - Multiple directories at once
✅ **Automation** - Headless mode with --dates
✅ **Version control friendly** - No local edits to Python files

## Migration Guide

### Old Workflow
1. Open `gamry_HiPOZOZ.py` in text editor
2. Find `dates = [...]` line
3. Edit to add your directory name
4. Save file
5. Run: `python gamry_HiPOZOZ.py`

**Problems:**
- Students might edit wrong part
- Merge conflicts in git
- Accidental code changes
- Requires understanding Python syntax

### New Workflow (GUI)
1. Run: `python gamry_HiPOZOZ.py`
2. Dialog opens
3. Click on your directory
4. Click "Select"

**Benefits:**
- No file editing
- No syntax errors
- Visual selection
- Works every time

### New Workflow (Command Line)
```bash
python gamry_HiPOZOZ.py --dates 20250815Mahboub2026
```

**Benefits:**
- One command
- Scriptable
- Fast
- Clear and explicit

## Technical Details

### Function: select_data_directories()

**Returns:** List of directory names (not full paths)

**Example:**
```python
# User selects: /path/to/data/20250815Mahboub2026
# Function returns: ['20250815Mahboub2026']
```

**Why just names?**
- Program expects directory names under `data/`
- Consistent with existing code structure
- Simpler path handling

### QFileDialog Configuration

```python
dialog = QFileDialog()
dialog.setFileMode(QFileDialog.Directory)      # Only directories
dialog.setOption(QFileDialog.ShowDirsOnly, True)  # Hide files
dialog.setWindowTitle("Select Data Directory(ies)")
dialog.setDirectory(start_dir)  # Start in data/
```

### Error Handling

**No directories selected:**
```
[INFO] User cancelled directory selection
[ERROR] No directories selected. Exiting.
```
- Exits with code 0 (user choice, not error)
- Clean shutdown

**Invalid directory:**
```
[ERROR] Data directory does not exist: data/InvalidName
```
- Logged and skipped
- Continues with other directories if multiple specified

## Testing

### Test 1: GUI Selection
```bash
$ python gamry_HiPOZOZ.py
[INFO] No data directories specified. Prompting user for selection...
[INFO] User selected directories: ['20250815Mahboub2026']
[INFO] Processing directories: ['20250815Mahboub2026']
✓ Success
```

### Test 2: Command Line
```bash
$ python gamry_HiPOZOZ.py --dates 20250815Mahboub2026
[INFO] Processing directories: ['20250815Mahboub2026']
✓ Success
```

### Test 3: Multiple Directories
```bash
$ python gamry_HiPOZOZ.py --dates 20250813 20250815
[INFO] Processing directories: ['20250813', '20250815']
✓ Success
```

### Test 4: Cancel Dialog
```bash
$ python gamry_HiPOZOZ.py
[INFO] No data directories specified. Prompting user for selection...
[INFO] User cancelled directory selection
[ERROR] No directories selected. Exiting.
✓ Exits gracefully
```

### Test 5: Headless Mode
```bash
$ python gamry_HiPOZOZ.py --headless --dates 20250815Mahboub2026
[INFO] HEADLESS ANALYSIS MODE
[INFO] Results saved to: data/20250815Mahboub2026/hipoz_20260318_results.csv
✓ Success
```

## Known Issues

### Issue 1: Config File Location for Non-Date Directories

**Problem:** Directories like `20250815Mahboub2026` create config in `data/20250815/` instead of `data/20250815Mahboub2026/`

**Cause:** Date extraction regex only captures 8 digits

**Status:** To be fixed

**Workaround:** Specify config path explicitly:
```bash
python gamry_HiPOZOZ.py --config data/20250815Mahboub2026/zAnalysis.csv
```

## Future Enhancements

### Planned Features
1. **Remember last directory** - Save user's last selection
2. **Recent directories** - Quick access to commonly used folders
3. **Drag & drop** - Drop folder onto script to analyze
4. **Config file templates** - Pre-populated templates for common setups
5. **Batch mode wizard** - GUI for setting up multiple analyses

### Under Consideration
- **Directory validation** - Check if directory contains valid data before accepting
- **Preview** - Show file count/dates before processing
- **Favorites** - Bookmark frequently used directories
- **Network paths** - Support for shared network drives

## Documentation Updates Needed

1. Update README with new workflow examples
2. Add screenshots of directory selection dialog
3. Create video tutorial for students
4. Update instructor's guide with batch processing examples

## See Also

- `HEADLESS_MODE.md` - Automated analysis documentation
- `FIXES_SUMMARY.md` - Recent bug fixes
- `GUI_CALIBRATION_DISPLAY.md` - GUI visualization features
