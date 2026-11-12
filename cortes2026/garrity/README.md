# Rose Garrity — benchtop data

Updated: 2026-11-05 (lab notebook received; see LABNOTES.md)

Rose Garrity took the run in `data/RoseData/Conductivity` (2026-02-19, ~296 K,
ambient pressure; 24 sweeps, triplets of each sample).

| Label | Content (from file labels) |
|---|---|
| `15mS`, `2764uS` | KCl conductivity standards (bottle labels, µS/cm) |
| `0.5MNaCl`, `0.75MNaCl`, `1MNaClRedo`, `1.5MNaCl`, `2MNaCl` | NaCl series |
| `Default` | unidentified |

`fit_rosedata.py` fits every sweep (CPE circuit, 10–100 kHz, as
`gamry_HiPOZ.py`) and writes `rosedata_20260219_fits.csv` with σ from each
standard's cell constant.

## Status: provisional

Rose's lab notebook (photos in `labnotes/`, transcribed in `LABNOTES.md`)
answers part of what was open:

1. **M is molal** (Steve, 2026-11-07: the lab prepares only molal
   solutions). Labels are mol/kg; no conversion.
2. **Cell-constant standard: still open.** Her probe reads both the 15 mS
   and 2764 µS bottles within 1% of nominal, so neither looks bad, yet the
   Gamry K from them differ by 3% (101.42 vs 98.40).
3. **`Default` triplet: not in the notebook.**
4. **2 M reads low:** still unexplained.

The notebook also identifies her NaCl:MgSO4 mixture runs with the Gamry
(data/20260618, 20260622, 20260624, by sweep time) and gives probe-only
mixture data at RT, 10 and 5 °C. Her Feb 5 probe readings seem to have had
temperature compensation on, so they are not raw 22.6 °C values.

Not yet merged into the Cortes compilation.
