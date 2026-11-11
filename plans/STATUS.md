# Status — hipozgenai

Updated: 2026-11-11

Refresh the `Updated:` line and the affected sections in any session that
pushes commits, integrates artifacts, or changes a queue.

## Current focus

Cortes 2026 paper: McCleskey/WATEQ4F model corrections — see audit 2026-10-05.
Previous: New sigma(P,T) GUI tab for diagnosing KCl-standard pressure trends; Sept 17-25
lab data merged in from origin/main. PR #3 (still open) now carries all of it.

## In flight

| Item | Owner | Status | Artifact |
|---|---|---|---|
| Merge origin/main lab data into hipozgenai | claude-code | verified | merge commit `eab7098`; audit entry 2026-09-14 in `coordination/audit.md` |
| Propagate hipozgenai to main | claude-code | verified | PR #2 merged by Steve 2026-09-15; merge commit `351853f` on `origin/main` contains branch tip `fd3718d` (`git merge-base --is-ancestor` confirmed) |
| Curate uncommitted working tree per Steve's decisions | claude-code | verified | commits `978540a` (tests/infra/docs), `e7b5f6f` (Aug 2025 Cortes + JesusCortes data), `af53f12` (mahboub2026 drafts); audit entry 2026-09-15 |
| New test suites pass in full project env (TeX + GUI) | claude-code | verified | Steve's macOS run 2026-10-05: 82 passed, 1 failed (the open KCl reference question), 9 skipped (Reaktoro absent in that env) |
| sigma(P,T) 2D map plot function (gamryPlots.plot_conductivity_PT) | claude-code | verified | commit `1f33a65`; tests/test_pt_map.py (8 passing); real-data map from 13 KCl dates sent to Steve 2026-09-15 |
| sigma(P,T) GUI tab in DataSelector | claude-code | implemented, unverified | commit `1f33a65`; container lacks libEGL - launch GUI on macOS and view the new tab |
| Merge origin/main Sept 17-25 data uploads | claude-code | verified | merge commit `58906cd`; `git merge-base --is-ancestor origin/main HEAD` passes; 248 files under data/20260917-20260925 |
| plot_sigma_PT.py CLI (map from curated/raw data) | claude-code | verified | tests/test_plot_sigma_PT.py (4 passing); full KCl run reproduces delivered figures in sigma_PT_plots/ |
| 3D sigma(P,T) surface (function + CLI output) | claude-code | verified | tests/test_pt_map.py 15/15 passing; sigma_PT_surface.png delivered 2026-10-03 |
| 3D sigma(P,T) GUI tab ("σ(P,T) 3D", drag-to-rotate) | claude-code | implemented, unverified | needs GUI launch on macOS (no libEGL in container) |
| sigma vs P isotherms (function + CLI output) | claude-code | verified | tests/test_pt_map.py 19/19; sigma_P_isotherms.png delivered 2026-10-03 |
| sigma vs P isotherms GUI tab ("σ(P) isotherms") | claude-code | implemented, unverified | needs GUI launch on macOS (no libEGL in container) |
| McCleskey total-ionic-strength fix | claude-code | verified | tests/test_mccleskey_ionic_strength.py (KCl standards within 1%); audit 2026-10-05 |
| WATEQ4F speciation in Cortes pipeline (auto mode) | claude-code | verified | regenerated cortes_plots/na2so4_vs_concentration.pdf (Reaktoro, macOS) matches independent PHREEQC 3.8.6 + wateq4f.dat at all 6 (T,m) points; audit 2026-10-08 |
| Cortes paper figures regenerated with corrected model | claude-code | verified | Steve ran regenerate_cortes_figures.sh 2026-10-08; committed; old figures in cortes2026/figures_archive_20261005_pre_mccleskey_fix/ |
| Speciation for mixtures (NaCl:MgSO4, Na2SO4:KCl) — speciation module | claude-code | verified | Steve's macOS run 2026-10-17: test_mixture_nacl_mgso4_matches_phreeqc, test_mixture_na2so4_kcl_matches_phreeqc, test_mixture_speciation_lowers_model pass (Reaktoro vs PHREEQC references) |
| Benchtop concentrations are molal as prepared (Steve 2026-11-07); 10-17 molarity conversion withdrawn | claude-code | verified | tests/test_cortes_concentrations.py, tests/test_mccleskey_validity.py; figures regenerated with Reaktoro (cortes2026/prepost_comparison/prepost_summary.txt; nacl_vs_concentration.pdf inspected) |
| Clean 0.5 m NaCl sigma(P) plot, unequilibrated near-room-T steps left out | claude-code | verified | cortes2026/nacl_sigma_P/nacl_0p5m_sigma_P.png inspected; exclusions in plot_nacl_sigma_P.py EXCLUDED; ~600 MPa 20 C points kept, boxed axes, 606 MPa 20 C point joined to 400 MPa; one panel with absolute sigma for 2026 NaCl (Oct 7-9, K from Oct 14 std) and KCl std (squares) and Pan et al. 2021 NaCl curves; Adams & Hall 1931 2.865 wt% NaCl at 25 C (R/R0 anchored to MC12 at 1 bar); Aug 18-Sep 10 KCl left out (Steve 2026-11-08) |
| Pan et al. (2021) NaCl conductivity (pan2021_nacl.py, ported from planetprofile-genai bf545b5e) | claude-code | verified | tests/test_pan2021_nacl.py (3 pass; reproduces the authors' 69 regression rows) |
| Supplement tables: Cortes, Diaz, Garrity (.tex, per-sweep verbatim + step means) | claude-code | verified | tests/test_supplement_tables.py (7 pass, every per-sweep cell equals its source string); cortes2026/supplement/preview.tex (supplement preamble, portrait) compiles with 0 overfull boxes; Steve to check data against sources |
| Mixture-family figures (all ratios/concentrations on one plot, vs T and vs m) | claude-code | verified | cortes2026/cortes_plots/mixture_families/*.png inspected; cortes2026/plot_mixture_families.py |
| Mixture ion_spec per first component (1.5-3x concentration error) | claude-code | verified | tests/test_cortes_concentrations.py; Na2SO4:KCl 1:1 RMS 34.2% -> 2.3% (prepost_summary.txt) |
| Pre/post-fix McCleskey comparison | claude-code | verified | cortes2026/prepost_comparison/ (PHREEQC backend; rerun with Reaktoro via regenerate_cortes_figures.sh) |
| Delta axis label states (data - model)/model | claude-code | verified | tests/test_delta_label.py; paper PDFs need regeneration on the Mac |
| Mixture speciation reaches paper figures (study_plots dropped speciation for ion_spec) | claude-code | implemented, unverified | tests/test_mccleskey_validity.py::test_mixture_figures_request_speciation passes; speciated mixture PDFs need Reaktoro run on the Mac (regenerate_cortes_figures.sh) |
| McCleskey validity limit (I_eff = 0.7 mol/kg) marked on paper figures | claude-code | verified | tests/test_mccleskey_validity.py (8 pass); container renders of NaCl, Na2SO4:KCl 1:1, NaCl:MgSO4 1:1 inspected; limits from cortes2026/mccleskey_validity_limits.py (PHREEQC wateq4f) |
| Merge origin/main Oct 5-15 2026 high-pressure uploads | claude-code | verified | merge commit `a8d2c12`; `git merge-base --is-ancestor origin/main HEAD` passes; compositions not recorded in files (all 'Default') - see open-questions 2026-10-19 |
| Aug 18-20 2025 Cortes NaCl 0.5 m high-pressure data in Cortes plots | claude-code | implemented, unverified | Aug 19-20 fitted (cortes2026/curate_aug2025_hp.py + headless --cal-map from Aug 18, K = 92.21); tests/test_cortes_loader.py passes; container render of NaCl_0.5M_vs_pressure inspected; paper PDFs need regenerate_cortes_figures.sh on the Mac |
| Import Z. Diaz zAnalysis workbooks (Sep 17/23/24/25, Oct 9 2026) | claude-code | verified | import_zanalysis_workbook.py; headless re-fit of Sep folders reproduces his R and sigma within 0.05% (108 measurements); configs gitignored (local) |
| Exclude Oct 9 run and end-of-Trial-2 triplet; document Diaz analysis | claude-code | verified | curate_diaz_2026.py; docs/DIAZ_2026_ANALYSIS.md; tests/test_curate_diaz_2026.py (4 pass); headless 20260923 analyses 45 of 49 rows and the config keeps all 49 rows, 14 columns, 3 excluded |
| Analysis never erases config contents: headless write-back, --cal-map, load-time CSV/JSON sync | claude-code | verified | config_store.py; tests/test_config_store.py (6 pass); real-data headless runs on 20260923 and 20260925 (--cal-map, stale JSON) changed only sigma and the borrowed row; docs/CONFIG_PRESERVATION.md |
| CSV is the source of truth; JSON only mirrors it (load-time sync, AnalysisConfig, config_store) | claude-code | verified | tests/test_config_store.py::test_csv_is_source_of_truth_at_load; real-data headless 20260923 with a corrupted newer JSON left the CSV byte-identical |
| Same, GUI save path | claude-code | implemented, unverified | tests/test_config_store.py::test_gui_save_merges_without_erasing calls save_gui_state_to_config directly; not yet exercised in the running GUI (no display/TeX here) |
| Chiller-bath P/T (config P_MPa = 0) overrides vessel header values | claude-code | verified | tests/test_bath_state.py (4 pass); headless re-fit reports bath setpoint T and P = 0 for all 108 Sep measurements; GUI path unverified on macOS |
| Mac run: full suite in `impedance` env (Reaktoro, PyQt6, TeX) | claude-code | verified | 2026-11-05: 135 passed, 1 failed (open KCl reference), 2 skipped (Reaktoro present); two tests chdir'd into a stale Dropbox copy, now repo-relative |
| Speciated mixture + Aug 2025 HP figures regenerated on the Mac | claude-code | verified | regenerate_cortes_figures.sh 2026-11-05 (Reaktoro); cortes2026/prepost_comparison/prepost_summary.txt; NaCl_0.5M_vs_pressure and nacl-mgso4_1-1 inspected |
| sigma(P,T), 3D, isotherm GUI tabs | claude-code | implemented, unverified | 2026-11-05 GUI launched on Aug 19 Cortes; log shows all three tabs lazy-loaded without error; visual check by Steve not yet recorded |
| GUI save path preserves config | claude-code | not implemented | 2026-11-05 GUI run turned the Aug 19 standard into a measurement, overwrote P/T with header values, rounded sigma; config restored from pre-GUI backup |
| GUI config discovery misses folder-suffixed configs (zAnalysis<date>Cortes.csv) | claude-code | not implemented | created a stray zAnalysis20250819.csv (moved to /tmp/gui_check/) |
| Cortes assessment: P < 10 MPa taken as 0.1 MPa (gauge value kept in P_MPa_raw) | claude-code | verified | cortes_data_processing.assess_low_pressure_as_ambient; tests/test_cortes_loader.py::test_low_pressure_readings_are_assessed_at_ambient; figures regenerated 2026-11-05 |
| Garrity RoseData (2026-02-19) provisional fit | claude-code | implemented, unverified | cortes2026/garrity/fit_rosedata.py; M = mol/kg (Steve 2026-11-07); cell-constant standard and Default triplet still open; not merged into Cortes |
| Lab notebooks transcribed (Diaz pp. 11-27, Garrity pp. 1-8) | claude-code | verified | cortes2026/diaz/LABNOTES.md, cortes2026/garrity/LABNOTES.md from labnotes/ photos |
| Nov 5 2026 run complete (51 sweeps); 5 fit-jump sweeps excluded (Steve) | claude-code | verified | data/20261105/zAnalysis20261105.csv (5 exclude=x, 51 rows kept); standard_sigma_P_vs_cortes.png inspected |
| Oct 14 124 MPa fit-jump triplet excluded (Steve); Oct 14/15 configs written | claude-code | verified | data/20261014/zAnalysis20261014.csv (3 exclude=x, 36 rows), 20261015 (15 rows); standard_sigma_P_vs_cortes.png inspected |
| Merge origin/main Nov 6 (Diaz's Windows run: PR #4 merged, data/20261106, Nov 5/6 configs now tracked) | claude-code | verified | merge commit `84412b3`; Nov 5 exclusions re-applied onto main's config (51 rows, 5 excluded); suite 138 passed / 1 known fail |
| Nov 6 2026 8 S/m KCl sigma(P), 294 K (Windows HiPOZ) | claude-code | verified | Mac refit of 24 sweeps matches Windows R within 0.03%; Windows used K = 120.54 (Nov 5); last 3 sweeps (175 MPa) added to config on Mac |
| main: GUI save erases config when headless falls back to GUI | claude-code | not implemented on main | reproduced on clean origin/main with 20261106 (config reduced to header); fixed on hipozgenai (config_store), PR #3 unmerged |
| KCl standard ppt basis = per kg water (Steve); 80 mS bottle Environmental Express 00653-32 | claude-code | verified | gamryTools.kcl_std_molal; tests/test_kcl_std_molality.py (3 pass); configs 20261014/15, 20261105/06 w_molal 0.699772 |
| KCl bottle ppt on water basis in GUI and Solution.CalcConc (ppt_to_molal/molal_to_ppt) | claude-code | verified | tests/test_kcl_std_molality.py (6 pass, incl. DataSelector.convert_loaded_concentrations); typing ppt in the live GUI not checked by hand |
| KCl fix to main | claude-code | implemented, unverified | PR #5 (branch `kcl-water-basis`, 115c8f2): code + Nov 5/6 w_molal; 5 KCl tests pass on the branch; awaiting merge |
| Merge origin/main Nov 6 decompression sweeps (18, 200->75 MPa) | claude-code | verified | merge commit 20e7c05; Nov 6 config 42 rows, all w_molal 0.699772; stage_data_changes.py arrives with it (auto `git add` of data/ while GUI runs) |
| Oct 5-9 identities | claude-code | verified | from Diaz notebook: Oct 5 80 mS KCl std 1 atm; Oct 6-9 ~5 wt% NaCl sigma(P) (cortes2026/diaz/LABNOTES.md) |
| Garrity NaCl:MgSO4 Gamry runs (Jun 18/22/24) identified, not analysed | claude-code | not implemented | sweep times in LABNOTES.md p. 7 |
| Contributor folders cortes2026/diaz, cortes2026/garrity | claude-code | verified | READMEs in each |
| Exclude noisy ~20 C NaCl 0.5 m days (Aug 19 at 20 C, Aug 14 600 MPa candidates) | claude-code | not implemented | awaiting Steve's choice of days |
| Merge origin/main Nov 5 2026 upload (data/20261105, 42 sweeps) | claude-code | verified | merge commit `98173d9`; `git merge-base --is-ancestor origin/main HEAD` passes |
| Gamry header dates in US style (11/5/2026) parse | claude-code | verified | gamryTools.parse_gamry_time; tests/test_gamry_time.py (3 pass); headless 20261105 processed 42 of 42 files |
| Nov 5 2026 8 S/m KCl standard sigma(P) (Diaz) fitted | claude-code | verified | data/20261105/zAnalysis20261105.csv (comp KCl, all rows); K = 120.54; unreliable sweeps kept per Steve |
| Diaz 8 S/m KCl sigma(P) (Oct 14-15, Nov 5) plotted alongside Cortes NaCl | claude-code | verified | cortes2026/diaz/standard_sigma_P_vs_cortes.png inspected; plot_standard_sigma_P.py |
| Diagnose unreliable sweeps (fit jumps at Oct 14 124 MPa, Nov 5 152-177 MPa) | claude-code | verified | cortes2026/diaz/refit_chained.py: chained initial guess removes all 20 jumps; sigma within 0.11% of phase-zero |Z|; refit_chained.png inspected |
| Production fitter: chained or multi-start initial guess | claude-code | not implemented | awaiting Steve's decision; fixed guess lands in a worse local minimum on ~20% of Oct 14-15 sweeps |
| McCleskey method write-up | claude-code | verified | docs/MCCLESKEY_METHOD.md; quoted lambda0(25 C) and ion-pair shares recomputed from code 2026-11-05; README limitation note replaced |
| Flexible Gamry date parser on main | claude-code | implemented, unverified | branch gamry-date-parser off origin/main (tests/test_gamry_time.py 3 pass there); PR awaiting merge |
| Stale pre-fix eis_plots not rebuilt by current script (4 PDFs) | claude-code | not implemented | see open-questions 2026-10-08 |

Status must be one of `verified` / `implemented, unverified` / `not implemented`.

## Blockers

- `test_kcl_extraction` fails on a reference-value question (55.88 vs 57.83
  mS/cm at 20C, 0.5 M KCl) — needs Steve's answer; see the 2026-09-15 entry in
  `coordination/open-questions.md`.
