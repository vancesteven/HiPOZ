# Status — hipozgenai

Updated: 2026-10-20

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
| Benchtop molarity -> molality conversion | claude-code | verified | tests/test_cortes_concentrations.py; cortes2026/molarity_to_molality.py (PHREEQC pitzer.dat densities) |
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
| Same, GUI save path | claude-code | implemented, unverified | tests/test_config_store.py::test_gui_save_merges_without_erasing calls save_gui_state_to_config directly; not yet exercised in the running GUI (no display/TeX here) |
| Chiller-bath P/T (config P_MPa = 0) overrides vessel header values | claude-code | verified | tests/test_bath_state.py (4 pass); headless re-fit reports bath setpoint T and P = 0 for all 108 Sep measurements; GUI path unverified on macOS |
| Stale pre-fix eis_plots not rebuilt by current script (4 PDFs) | claude-code | not implemented | see open-questions 2026-10-08 |

Status must be one of `verified` / `implemented, unverified` / `not implemented`.

## Blockers

- `test_kcl_extraction` fails on a reference-value question (55.88 vs 57.83
  mS/cm at 20C, 0.5 M KCl) — needs Steve's answer; see the 2026-09-15 entry in
  `coordination/open-questions.md`.
