# AGU 2026 poster: facts and figures

Everything here is computed from the existing CSVs in `../nacl_sigma_P/` using the main figure's run
selection, exclusions and anchoring (`../plot_sigma_P_main.py`, `../sigma_P_sources.py` are imported, not
modified). No re-fitting. Run from the repo root (PlanetProfile may prompt; `echo n |` answers it):

    echo n | python cortes2026/agu2026_poster/poster_facts.py          # poster_facts.md, poster_facts.csv
    echo n | python cortes2026/agu2026_poster/make_poster_figures.py   # fig_*.png / fig_*.pdf

No LaTeX needed (mathtext, DejaVu Sans). Use MPLBACKEND=Agg.

Inputs: nacl_sigma_P/nacl_0p5m_sigma_P_steps.csv, sigma_P_2026_steps.csv, kcl_literature_ratio.csv,
literature/adams_hall_1931_nacl_2p865.csv, pan2021_nacl.py, vessel_1bar_check.py (1-bar reference).

| Figure | File stem | Print size | Pixels (300 dpi) |
|---|---|---|---|
| A: sigma vs P, 3 panels | fig_A_sigma_P | 20 x 7.5 in | 6000 x 2250 |
| B: sigma(P)/sigma(1 bar), NaCl 0.9, with Adams & Hall and Pan | fig_B_relative_sigma_0p9m | 9 x 7.5 in | 2700 x 2250 |
| C: % increase to 200 MPa vs T | fig_C_pct_increase_vs_T | 7 x 6 in | 2100 x 1800 |

Figures are drawn at print size, so font sizes in the script are the sizes on the poster: axis labels 22 pt,
ticks 18 pt, titles 24 pt, direct labels 18 pt (17 pt in B), footnotes 16 pt (14.5 pt in B, C), lines 3 pt, markers 10.5 pt.
Colour = temperature on one continuous blue-to-red scale (not the main figure's TEMP_COLORS bins, which are not
monotone). Filled = compression, open = decompression. Fig C marks the 9.85 C Aug-2025 point "1-bar step
unsettled?" (set `FLAG_10C = False` to remove; see poster_facts.md section h).
