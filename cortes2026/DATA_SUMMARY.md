# Cortes 2026 - Data Organization Summary

## ✅ Completed: Benchtop Data Parsing

Successfully converted **JesusData2025.csv** from spreadsheet format to standardized **Cortes2026BenchtopData.csv**.

### Data Overview

- **Total measurements**: 132
- **Compounds**: 13
- **Temperature range**: 5-20°C  
- **Concentration range**: 0.1-1.5 M

### Compounds Included

| Compound | N | Temps | Concs | Notes |
|----------|---|-------|-------|-------|
| **Salts** |
| KCl | 12 | 3 | 4 | 0.5-1.5 M/L |
| NaCl | 24 | 1 | 4 | 0.5-1.5 M/L, 6 replicates each |
| Na₂SO₄ | 6 | 3 | 2 | 0.5-0.75 M/L |
| **Salt Mixtures** |
| NaCl:MgSO₄ (1:1) | 12 | 3 | 4 | 0.5-1.5 M/L each |
| NaCl:MgSO₄ (2:1) | 3 | 3 | 1 | 1.5:0.75 M/L |
| NaCl:MgSO₄ (1:2) | 3 | 3 | 1 | 0.75:1.5 M/L |
| Na₂SO₄:KCl (1:1) | 9 | 3 | 3 | 0.5-1.0 M/L each |
| Na₂SO₄:KCl (2:1) | 3 | 3 | 1 | 1.5:0.75 M/L |
| **Organic Compounds** |
| NaCl + Glycine | 24 | 1 | 4 | NaCl 0.5-1.5 M + 100mM glycine, 6 replicates |
| Glycine | 9 | 1 | 1 | 100 mM/L, 9 replicates |
| Alanine | 9 | 1 | 1 | 100 mM/L, 9 replicates |
| Aspartic Acid | 9 | 1 | 1 | 100 mM/L, 9 replicates |
| Glutamic Acid | 9 | 1 | 1 | 100 mM/L, 9 replicates |

### Key Findings

**Glycine Effect on NaCl Conductivity (20°C):**
- 0.5 M NaCl: +1.1% increase with 100 mM glycine
- 1.5 M NaCl: +0.5% increase with 100 mM glycine
- **Conclusion**: Glycine has minimal impact on conductivity (doesn't significantly dissociate)

**Amino Acid Conductivities (100 mM/L, 20°C):**
- Glycine: 1.02 ± 0.12 S/m (zwitterion, minimal conductivity)
- Alanine: 1.02 ± 0.29 S/m (zwitterion, similar to glycine)
- **Aspartic Acid: 84.25 ± 1.14 S/m** (dicarboxylic! very high)
- **Glutamic Acid: 55.22 ± 1.37 S/m** (dicarboxylic! very high)

The dicarboxylic amino acids (aspartic and glutamic acid) show extraordinarily high conductivities because they can donate multiple protons and exist in highly charged forms.

### Data Format

File matches **Mahboub2026BenchtopData.csv** format:
```
compound, concentration_molal, temperature_C, temperature_K,
conductivity_Sm, replicate, source, notes
```

**Units:**
- Concentration: mol/kg (molal) - *Note: converted from M/L assuming dilute solutions*
- Temperature: °C and K
- Conductivity: S/m - *Converted from mS/cm (÷10)*

### Data Quality Notes

1. **Concentration units**: Original data in M/L (molar), converted to molal assuming ρ≈1 for dilute solutions. May need refinement for accurate thermodynamic modeling.

2. **Multiple replicates**: 
   - NaCl: 6 replicates per condition (excellent!)
   - Amino acids: 9 replicates per condition (excellent!)
   - Salt mixtures: 1 replicate per condition

3. **Temperature coverage**: Most compounds measured at 5°C, 10°C, 20°C. NaCl and organics only at 20°C.

## 📋 Next Steps

### Task #2: Organize Gamry High-Pressure Data

**Current location**: `../JesusCortes/Data/`  
**Date folders**: 08.12.25, 08.13.25, 08.14.25, 08.15.25

**Gamry data includes:**
- Calibration standards: 15 mS/cm, 80 mS/cm
- NaCl pressure series: 0.5, 0.75, 1.0, 1.5, 2.0 M
- NaCl + MgSO₄ mixtures
- **Pressure range**: 0-600 MPa (!)
- **Temperature**: ~20°C (room temp)

**Actions needed:**
1. Copy/organize Gamry .txt files into `data/20250813Cortes/`, etc.
2. Run HiPOZ impedance analysis pipeline
3. Generate `hipoz_*_results.csv` files
4. Integrate with benchtop data in plotting script

**Status**: ⏸️ Waiting for user instructions
