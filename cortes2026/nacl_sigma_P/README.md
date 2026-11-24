# sigma(P) figures: pipeline

Two commands regenerate everything (run from the repository root, env `impedance`):

```
python cortes2026/plot_nacl_sigma_P.py   # ~50 s: fits all sweeps, writes CSVs + SUPPLEMENT figure
python cortes2026/plot_sigma_P_main.py   # seconds: MAIN-TEXT figure from the CSVs
```
(`plot_nacl_sigma_P.py --figures-only` redraws the supplement from existing CSVs in seconds.)

## Order and files

| Step | Script | Writes | Feeds |
|---|---|---|---|
| 1 | `plot_nacl_sigma_P.py` | `nacl_0p5m_sigma_P_steps.csv` (Aug 2025 steps, `excluded` column), `sigma_P_2026_steps.csv` (2026 kept steps), `../diaz/diaz_hp_sweep_fits.csv` (all 2026 sweeps, exclusion reasons), `kcl_literature_ratio.csv` | `sigma_P_sources.py`, `supplement_tables.py` |
| 2 | `sigma_P_sources.py` (module, no plotting) | none | both figures; data model in its docstring |
| 3 | `plot_nacl_sigma_P.py` | `sigma_P_supplement.{pdf,png}` (+ deprecated duplicate `nacl_0p5m_sigma_P.{pdf,png}`) | supplement |
| 4 | `plot_sigma_P_main.py` | `sigma_P_main.{pdf,png}`, `sigma_P_main_anchor_factors.csv` | main text |

Supplement: columns NaCl | KCl; top absolute sigma (S/m, KCl log axis), bottom sigma/sigma(first 1-bar step).
Shows everything: excluded steps (black x), decompression (hollow), 1-bar checks, Pan 2021, Adams & Hall, Fisher & Fox, Fisher 1962.
The former `nacl_0p5m_sigma_P_excluded.png` was removed: its excluded steps are now in the supplement (times in the CSVs).

## 1-bar reference hierarchy (`sigma_P_sources.sigma_ref_1bar`)
1. McCleskey (2011) Table 3 measurements, 0.5-1.0 mol/kg and 5-90 C (NaCl: `vessel_1bar_check.sigma_prior_nacl`; KCl: same method). KCl table values are transcribed from conductivity-jpl, not yet checked against the paper.
2. McCleskey et al. (2012) model elsewhere (e.g. below 5 C).

## Anchoring policy (main figure, `ANCHOR = True`)
Each run is multiplied by (reference sigma at the run's first ascending 1-bar step) / (measured sigma there). The measured pressure dependence is kept; the run's own absolute calibration is replaced by prior 1-bar data ("trust the composition, calibrate the 1-bar point against prior data", PI 2026-11-18, applied to Nov 16). Assumed anchor uncertainty: 1 % (McCleskey 2011), 2 % (MC12): for PI review. Factors: `sigma_P_main_anchor_factors.csv`.

## Main-figure runs (`MAIN_RUNS`)
Aug 2025 0.5 m NaCl kept steps; NaCl 0.9 m Oct 7-8, Nov 16 and the six Nov 17-19 isotherms (24.8, 20.5, 16.2, 11.5, 7.0, ~2 C; boundaries `RUN_WINDOWS` in `plot_nacl_sigma_P.py`); KCl 0.70 m Nov 5, 6, 9, 10, 11, 11-12.
Not shown: Oct 6, Oct 9 (10-min steps), Oct 14-15 (leak during run), excluded/provisional steps, Aug-Sep 2026.

## Exclusions
2025: `EXCLUDED` in `plot_nacl_sigma_P.py` (reason per first-sweep time). 2026: zAnalysis config `exclude` flags plus `PROVISIONAL_2026` (Nov 12 +/-12 ohm CPE fits). Nothing is removed from the configs.

## Nov 17-19 2026 (Diaz, 0.902 mol/kg NaCl)
- Cell constant: the potentiostat connection was repaired, so K is re-set from the data (PI 2026-11-20, "yes to the self calibration"). Supplement: ONE K (120.093 1/m) for all six isotherms, from the first 1-bar step of the 24.8 C isotherm (Nov 17 09:46) vs McCleskey (2011); the colder isotherms' 1-bar offsets from the reference then remain visible as a diagnostic. Main figure: each isotherm anchored at its own first ascending 1-bar step; the ~2 C isotherm uses MC12 (below 5 C), recorded in the factors table.
- Out of the vessel: Nov 19 from 16:44 (sweeps 164412 ... 175155) is a KCl 8 S/m calibration in a separate chiller; the logged T is the vessel sensor. These 12 sweeps are in `diaz/diaz_hp_sweep_fits.csv` with an exclusion reason (the 17:50 ones, R 10.1-10.4 ohm drifting, also "suspected bad wire orientation") and no sigma; they are not part of any isotherm.
- Colours: `TEMP_COLORS` gained a 4-8 C bin (violet) so 2 C and 7 C differ; the palette was not run through the dataviz validator (not installed).
