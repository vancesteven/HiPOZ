# Multi-Component Concentration Handling

The GUI now supports multi-component solutions with automatic unit conversion between ppt and molal.

## Format

### Single Component
- **Comp**: `NaCl`
- **w (ppt)**: `55.21` (float)
- **w (molal)**: `1.0` (float)

### Multi-Component
- **Comp**: `NaCl,MgSO4` (comma-separated)
- **w (ppt)**: `137.84` (total dissolved solids - single value)
- **w (molal)**: `1.5,0.6` (individual molalities - comma-separated)

## Usage

### Single Component
1. Enter compound name in Comp (e.g., `NaCl`)
2. Enter either ppt or molal (single number)
3. The other field is automatically calculated

### Multi-Component
1. Enter comma-separated compound names in Comp (e.g., `NaCl,MgSO4`)
2. Enter comma-separated molal values (e.g., `1.5,0.6`) - **molal is preferred for mixtures**
3. Total ppt is automatically calculated as a single value (e.g., `137.84`)

## Supported Compounds

- `NaCl` (58.44 g/mol)
- `KCl` (74.55 g/mol)
- `MgSO4` (120.37 g/mol)

Add more compounds to `MOLAR_MASSES` dictionary in `hipoz_data_selector_gui.py`.

## Conversion Formulas

### Multi-Component: Molal → ppt

For multi-component solutions, ppt is the **total dissolved solids**:

w_ppt = (m_solute1 + m_solute2 + ...) / (1000g H2O + m_solute1 + m_solute2 + ...) × 1000

where m_solute is calculated from molality: m_i = b_i × M_i

**Example**: NaCl,MgSO4 at 1.5,0.6 molal
- NaCl mass = 1.5 mol/kg × 58.44 g/mol = 87.66 g
- MgSO4 mass = 0.6 mol/kg × 120.37 g/mol = 72.22 g
- Total solute mass = 87.66 + 72.22 = 159.88 g
- Total solution mass = 1000 + 159.88 = 1159.88 g
- **w_ppt = (159.88 / 1159.88) × 1000 = 137.84**

### Multi-Component: ppt → Molal

**Note**: Converting from total ppt to individual molalities is ambiguous without knowing the composition ratio. For multi-component solutions, **use molal as the primary input format**.

If individual ppt values are available (rare), the conversion is:

1. Total ppt = Σ(ppt_i)
2. Water mass = (1000 - total ppt) / 1000 kg
3. Molal_i = (ppt_i / M_i) / water_mass

**Why molal is preferred for mixtures**: When preparing a solution, you add specific amounts of each solute (measured in moles), making molality the natural unit. The total ppt is then a derived quantity.

## Config File Format

Both CSV and JSON formats support multi-component concentrations:

### CSV
```csv
group_name,filename,P_MPa,T_K,type,Z_Ohm,Z_unc_Ohm,conductivity_Sm,S_unc_pct,comp,w_ppt,w_molal,exclude,notes
Group 1,file.txt,100,273,measurement,,,,"NaCl,MgSO4",137.84,"1.5,0.6",,
```

**Note**: For multi-component solutions, w_ppt is the total dissolved solids (single value), while w_molal lists individual molalities (comma-separated).

### JSON
```json
{
  "calibrations": [
    {
      "measurements": [
        {
          "filename": "file.txt",
          "comp": "NaCl,MgSO4",
          "w_ppt": 137.84,
          "w_molal": "1.5,0.6",
          "P_MPa": 100,
          "T_K": 273
        }
      ]
    }
  ]
}
```

**Note**: w_ppt is a number (total), w_molal is a string (comma-separated individual values).

## Notes

- Values must have the same number of comma-separated entries
- Spaces around commas are automatically trimmed
- Mixing single and multi-component formats in the same dataset is supported
- Conversion only occurs when all compounds are recognized in the molar mass dictionary
