# Sep–Oct 2026 runs: Zachary Diaz's analysis, import and exclusions

Updated: 2026-10-20

Zachary Diaz took and analysed the Sep–Oct 2026 data. His zAnalysis tables
and notes are in `labnotes/*.xlsx` (one workbook per run), and his
KCl-standard σ(P) plot is
`labnotes/KCl Standard - Conductivity vs Pressure at Room Temp.png`. This note records what the runs are, how they
were imported, what was checked, what was excluded and why, and what is still
open. Open items are tracked in `coordination/open-questions.md`
(2026-10-19 and 2026-10-20 entries).

## Runs

| Folder | Workbook | Solution | Conditions | Standard |
|---|---|---|---|---|
| `data/20260917` | 2026.09.17 … Trial 1 | 0.5 m NaCl (28.39 ppt) | 1 bar; bath −1, 5, 20, 25 °C | 8 S/m, Z = 14.95 Ω |
| `data/20260923` | 2026.09.23 … Trial 2 (Pre vs Post Gamry Calibration) | 0.5 m NaCl | 1 bar; bath cycle 25 → −1 → 25 °C | 8 S/m, Z = 14.99 Ω (taken before the recalibration) |
| `data/20260924`, `data/20260925` | 2026.09.24 … Trial 3 (Post Gamry Calibration) | 0.5 m NaCl | 1 bar; bath cycle over two days | 8 S/m, Z = 14.83 Ω (Sep 24; also used for Sep 25) |
| `data/20261009` | 2026.10.09 5 wt % NaCl Conductivity vs Pressure | 5 wt% NaCl (50.021 ppt, 0.901 m) | 22.8 °C; 0 → 200 → 0 MPa | none in workbook |

Also identified from his files but not imported: the Oct 14–15 KCl-standard
σ(P) sweep (the PNG; 23.1–23.5 °C, 8.0 → 8.55 S/m from 0 to 200 MPa, with
one 124 MPa triplet at 8.97 S/m as an outlier). Oct 5–8 are still
unidentified.

## Import

`import_zanalysis_workbook.py labnotes/*.xlsx` writes one
`data/<date>/zAnalysis<date>.csv` (plus JSON) per day:

- Columns A–N of his sheet are kept as is, including his Z_Ohm, Z_unc_Ohm and
  S_unc_pct.
- Rows are split by the date in the Gamry filename, so the Trial 3 workbook
  becomes the Sep 24 and Sep 25 configs. The Sep 25 rows note that the Sep 24
  standard calibrates them.
- Every row's notes name the source workbook. The Trial 2 rows also say
  "before" or "after Gamry recalibration", taken from his red/blue font
  colours.
- Oct 9 was imported with `--note "no standard in workbook; implied K=100.2
  vs ~120 in Sept; sigma ~20% below CRC 5 wt% NaCl (open question
  2026-10-19)"`.

The configs are gitignored, so they are local. To rebuild them, run the
import, then `python curate_diaz_2026.py` (below).

## Chiller-bath P and T

At 0 MPa the sample holder sits in the PolyCool chiller bath, so the config's
T and P are the bath values (Steve, 2026-10-19). The Gamry file headers carry
the vessel readings instead (about 300 K and 2 MPa throughout these runs).
Config rows with P_MPa = 0 now override the header in the headless and GUI
paths (`analysis_config.bath_state`), unless the header shows the vessel
above 5 MPa.

## Check against his numbers

A headless re-fit of the four September folders from the raw spectra
reproduces his R and σ within 0.05% for all 108 measurements. Cell
constants: 119.60 (Sep 17), 119.89 (Sep 23), 118.62 (Sep 24; also used for
Sep 25). After the bath fix, the headless output reports his bath setpoints
and P = 0.

## Results (exclusions applied)

Mean σ in S/m at each bath setpoint (n = number of readings). "Lit" is the
reference column in his workbooks (2.699, 3.134, 4.431, 4.874 S/m). It is not
McCleskey 2012, and its source is unknown.

| Run | −1 °C | 5 °C | 20 °C | 25 °C |
|---|---|---|---|---|
| Trial 1 (9/17) | 1.849 (3), −31% | 2.516 (6), −20% | 4.142 (6), −6.5% | 4.691 (3), −3.7% |
| Trial 2 before recal (9/23) | 2.705 (6) | 3.197 (9) | 4.455 (6) | 4.982 (3) |
| Trial 2 after recal (9/23) | 2.731 (6) | 3.225 (6) | 4.615 (6) | 5.079 (3) |
| Trial 3 (9/24) | 2.495 (6), −7.5% | 2.932 (9), −6.4% | 4.121 (6), −7.0% | 4.762 (6), −2.3% |
| Trial 3 (9/25) | — | 2.996 (3), −4.4% | 4.206 (6), −5.1% | 4.931 (6), +1.2% |
| His "Lit" | 2.699 | 3.134 | 4.431 | 4.874 |

The percentages are deviations from "Lit". Trial 2 as a whole runs +0.7%,
+2.4%, +2.3% and +3.2% above "Lit" at −1, 5, 20 and 25 °C.

Observations:
- **Trial 2 after the recalibration** reads 0.9–3.6% higher than before it at
  every setpoint. The only standard was measured before the recalibration,
  so the "after" values use a pre-recalibration cell constant.
- **Trials disagree.** Trials 1 and 3 sit below "Lit" and Trial 2 sits above
  it. At 20 °C the spread is about 11% (4.12–4.62 S/m), much larger than the
  ±0.1–0.4% fit uncertainties.
- **Trial 1 at 5 °C** contains one triplet at 2.963 S/m (taken before the
  −1 °C step) and one at about 2.07 S/m (taken after it). The low −1 °C value
  and the low 5 °C triplet after it fit partial freezing at −1 °C, but nothing
  confirms it.

## Exclusions

`curate_diaz_2026.py` applies these, sets `exclude = x` and appends the
reason to each row's notes. Running it again changes nothing. Excluded rows
stay in the config (CSV and JSON) and are skipped by the headless and GUI
analyses.

| Folder | Rows | Reason |
|---|---|---|
| `20260923` | `Default_20260923_174008`, `_174102`, `_174155` (3 readings, end of Trial 2, 25 °C) | Z = 45.3–47.8 Ω against 23.6 Ω minutes earlier at the same setpoint, so σ is about 50% low (2.51–2.65 S/m). The jump, and the scatter within the triplet, point to a measurement fault, not the solution. |
| `20261009` | all 45 readings (5 wt% NaCl σ(P)) | σ = 5.945 S/m at 22.8 °C and 0 MPa is about 20% below literature (CRC: 7.01 S/m at 20 °C; the Cortes benchtop NaCl interpolates to about 7.3). The workbook has no standard row, and his σ implies K = 100.2 against about 119.6 in September. Until the cell constant is resolved, the absolute σ is not usable. The pressure dependence (+7.3% from 0 to 200 MPa, reversible within 0.1%) may still be useful if K is constant within the run. |

Steve decided both exclusions on 2026-10-20. Trial 1 at −1 °C and the low
5 °C triplet after it are **not** excluded.

## Pipeline fixes made along the way

Two config round trips lost data. Both are covered by tests in
`tests/test_curate_diaz_2026.py`.
- **Headless write-back.** It rewrote CSV configs from the loaded
  calibration groups. The CSV loader drops excluded rows, so every headless
  run deleted the excluded rows, their reasons, and any extra columns.
  `headless_analysis._update_config_with_conductivities` now updates the CSV
  in place and regenerates the JSON from it.
- **CSV ↔ JSON harmonization.** `harmonize_config`, which runs automatically
  at load, wrote only the 11 standard columns. Extra columns such as Z_Ohm
  now round-trip.

JSON → CSV harmonization still blanks measurement σ, as before. For excluded
rows, his σ therefore survives only in the workbook, while his Z stays in the
config.

## Open

1. **Oct 9 cell constant.** Which standard set K = 100.2, and did the cell
   change between September and October?
2. **"Lit" values.** Where do they come from?
3. **Trial-to-trial spread.** The trials disagree by up to about 11%, and
   Trial 2 shifts after the recalibration. The cause is unknown. Candidates
   are the cell constant per trial, a single pre-recalibration standard, and
   bath versus sample temperature.
4. **Oct 5–8 runs.** Not yet identified.
