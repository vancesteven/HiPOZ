# GUI Calibration Display

## New Feature: Calibration Column

The DataSelector GUI now includes a **Calibration** column that clearly shows which standards are associated with which measurements.

## What You'll See in the GUI

### Table Structure

The table now has these columns:
1. **Filename** - Data file name
2. **Calibration** ← NEW! Shows calibration associations
3. **Time** - Measurement timestamp
4. **Comp** - Composition (KCl, NaCl, etc.)
5. **w (ppt)** - Concentration
6. **T (K)** - Temperature
7. **P (MPa)** - Pressure
8. **Z (Ohm)** - Measured resistance
9. **Z± (Ohm)** - Resistance uncertainty
10. **S (S/m)** - Computed conductivity
11. **S± (S/m)** - Conductivity uncertainty

### Visual Indicators

#### Row Colors
- **Light Blue Rows** → Standards (calibration references)
- **Light Green Rows** → Measurements (samples being analyzed)

#### Calibration Column Content

**For Standards (Light Blue Rows):**
```
Group 1 [Standard]
```
- Shows this file is a calibration standard
- Part of "Group 1" calibration group

**For Measurements (Light Green Rows):**
```
Group 1 [→ 3 stds: 0, 1, 2]
```
- Shows this measurement uses standards from "Group 1"
- Arrow (→) indicates association direction
- "3 stds" means 3 standards were used for calibration
- "0, 1, 2" are the row indices of those standards

### Example Display

For your Mahboub data (20250813Mahboub2026), you should see:

| Row | Filename | Calibration | Comp | w (ppt) | P (MPa) | S (S/m) | Color |
|-----|----------|-------------|------|---------|---------|---------|-------|
| 0 | Default_20250813_111634_P_0_T_0.txt | **Group 1 [Standard]** | KCl | | 0 | | 🔵 Blue |
| 1 | Default_20250813_111727_P_0_T_0.txt | **Group 1 [Standard]** | KCl | | 0 | | 🔵 Blue |
| 2 | Default_20250813_111820_P_0_T_0.txt | **Group 1 [Standard]** | KCl | | 0 | | 🔵 Blue |
| 3 | Default_20250813_115907_P_0_T_0.txt | **Group 1 [→ 3 stds: 0, 1, 2]** | NaCl | 1.0 | 0 | 8.281 | 🟢 Green |
| 4 | Default_20250813_120000_P_0_T_0.txt | **Group 1 [→ 3 stds: 0, 1, 2]** | NaCl | 1.0 | 0 | 8.283 | 🟢 Green |
| 5 | Default_20250813_120053_P_0_T_0.txt | **Group 1 [→ 3 stds: 0, 1, 2]** | NaCl | 1.0 | 0 | 8.278 | 🟢 Green |
| 6 | Default_20250813_135054_P_3_T_293.txt | **Group 1 [→ 3 stds: 0, 1, 2]** | NaCl | 0.5 | 3 | 4.700 | 🟢 Green |
| 7 | Default_20250813_135147_P_2_T_293.txt | **Group 1 [→ 3 stds: 0, 1, 2]** | NaCl | 0.5 | 2 | 4.696 | 🟢 Green |
| 8 | Default_20250813_135241_P_3_T_293.txt | **Group 1 [→ 3 stds: 0, 1, 2]** | NaCl | 0.5 | 3 | 4.690 | 🟢 Green |

### Reading the Calibration Associations

**Easy interpretation:**
1. **Rows 0-2** are standards (blue) - these are your KCl calibration files
2. **Rows 3-8** are measurements (green) - these are your NaCl samples
3. Each measurement shows "→ 3 stds: 0, 1, 2" meaning:
   - It was calibrated using 3 standards
   - Those standards are in rows 0, 1, and 2
   - You can visually trace which standards were used

### Multiple Calibration Groups

If you have multiple groups (e.g., for drift correction), you'll see:

| Row | Filename | Calibration | Notes |
|-----|----------|-------------|-------|
| 0 | std_morning_1.txt | **Group 1 [Standard]** | Morning calibration |
| 1 | std_morning_2.txt | **Group 1 [Standard]** | |
| 2 | sample_1.txt | **Group 1 [→ 2 stds: 0, 1]** | Uses morning standards |
| 3 | sample_2.txt | **Group 1 [→ 2 stds: 0, 1]** | Uses morning standards |
| 4 | std_afternoon_1.txt | **Group 2 [Standard]** | Afternoon recalibration |
| 5 | std_afternoon_2.txt | **Group 2 [Standard]** | |
| 6 | sample_3.txt | **Group 2 [→ 2 stds: 4, 5]** | Uses afternoon standards |
| 7 | sample_4.txt | **Group 2 [→ 2 stds: 4, 5]** | Uses afternoon standards |

**Clear separation:**
- Group 1: Rows 0-3 (morning calibration)
- Group 2: Rows 4-7 (afternoon recalibration)
- Each measurement explicitly shows which standards it uses

## Interactive Features

### 1. Hover Tooltips
- Hover over the **Filename** column (first column)
- See tooltip: "Standard: Group 1" or "Measurement: Group 1"

### 2. Visual Tracing
- Click on a green (measurement) row
- Look at its "Calibration" column to see which standards were used
- Those standard rows are highlighted in blue
- You can visually verify the calibration chain

### 3. Sorting
- Click column headers to sort
- Sort by "Calibration" to group all standards together, then measurements

### 4. Plotting
- Select multiple rows (Ctrl+Click or Shift+Click)
- Click "Create Plots" button
- View impedance spectra for selected files
- Compare standards vs measurements visually

## Benefits

### Before (Without Calibration Column):
- ❓ Which standards were used for this measurement?
- ❓ Are these files related?
- ❓ Manual inspection of CSV needed

### After (With Calibration Column):
- ✅ Instant visual indication: "Group 1 [→ 3 stds: 0, 1, 2]"
- ✅ Clear standard→measurement associations
- ✅ Multiple groups easily distinguished
- ✅ Traceable calibration workflow

## For Students

**Quick workflow verification:**
1. Look for blue rows → These are your standards
2. Look for green rows → These are your samples
3. Read the "Calibration" column:
   - See "→ 3 stds" → Confirms 3 standards were used
   - See "0, 1, 2" → Those specific standards
4. Verify it makes sense for your experimental setup

**Common patterns:**
- All measurements use same standards → Single calibration group
- Some measurements use different standards → Multiple groups (drift correction)
- Standard at beginning and end → Bracketing calibration

## Implementation Details

### How It Works
1. CSV/JSON specifies `group_name` for each file
2. Specifies `type`: "standard" or "measurement"
3. GUI groups files by `group_name`
4. Computes cell constant from standards in each group
5. Applies to measurements in same group
6. Displays associations in Calibration column

### Column Generation
- Generated automatically when config is loaded
- Updated when calibration is applied
- Persists through GUI session
- Not saved to CSV (derived from CSV structure)

### Color Coding Algorithm
```python
For each calibration group:
  For each standard in group:
    - Set row color to light blue
    - Set Calibration column: "Group X [Standard]"

  For each measurement in group:
    - Set row color to light green
    - Set Calibration column: "Group X [→ N stds: indices]"
    - Apply cell constant from standards
```

## Troubleshooting

### "Calibration column is empty"
**Cause:** No CSV/JSON config loaded, or GUI not started with `--gui`

**Fix:**
1. Ensure `zAnalysis<date>.csv` exists in data directory
2. Launch with: `python gamry_HiPOZ.py --gui`

### "No color highlighting"
**Cause:** Config loaded but calibration not applied

**Fix:** Check for error messages in terminal. Config must have:
- At least one standard with `conductivity_Sm` value
- At least one measurement

### "Wrong standards shown in Calibration column"
**Cause:** CSV has incorrect `group_name` associations

**Fix:** Edit CSV to assign correct `group_name` to each file:
```csv
group_name,filename,type,...
Group 1,std1.txt,standard,...
Group 1,sample1.txt,measurement,...
Group 2,std2.txt,standard,...
Group 2,sample2.txt,measurement,...
```

## See Also

- `HEADLESS_MODE.md` - Automated analysis without GUI
- `FORMAT_HARMONIZATION.md` - CSV/JSON format details
- `analysis_config.py` - Config file structure
