# Rose Garrity — benchtop data

Updated: 2026-11-05

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

Waiting on Rose's lab notebook for:

1. **M = molar or molal?** Against McCleskey (2012) the two readings differ by
   only 1–4% at 0.5–1.5, so the data cannot decide it.
2. **Which standard sets the cell constant.** The two disagree by 3%
   (K = 101.42 from 15 mS/cm, 98.40 from 2764 µS/cm).
3. **What the `Default` triplet is.**
4. **Whether the 2 M sample reads low** (8–11% below McCleskey as molal).

These data are not yet merged into the Cortes compilation. Once the notebook
answers the above, NaCl rows at compositions the Cortes series also covers
will be added there.
