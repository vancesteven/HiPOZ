# Audit log — hipozgenai

Append-only. One entry per completed action, newest at the bottom. Never edit
or delete existing entries. Format is defined in `AGENTS.md`.

## 2026-09-14 — claude-code — merge origin/main data into hipozgenai

- Merged `origin/main` into `hipozgenai` (merge commit `eab7098`).
- Brought in 665 new files, all lab-data text files (July 26 – Sept 11, 2026:
  `data/<date>/` Gamry sweeps, PressTemps, calibration logs). No Python code
  changed by the merge — main's only code edits were the matplotlib
  `get_cmap` fix, already present here from `16a66ed`, so both sides
  auto-merged identically. PyQt6 migration on this branch is intact.
- Note: main carries five data folders at the repo root (`20260801/` –
  `20260805/`) instead of under `data/`; inherited as-is, not moved.
- Status: verified (merge commit inspected; `git diff fa82535 HEAD
  --name-status -- '*.py'` empty; PyQt6 + colormaps imports confirmed in
  `gamry_HiPOZ.py`, `gamryTools.py`, `Pan_HiP.py`, `study_plots.py`).

## 2026-09-14 — claude-code — relocate root-level data folders into data/

- Moved `20260801/`–`20260805/` (PressTemps + calibration logs, inherited from
  main's upload) from the repo root to `data/<date>/`, matching every other
  dated data folder. `git mv` only; file contents unchanged.
- Same fix to be applied on `main`, per Steve's request.
- Status: verified (git status shows pure renames; no content diff).

## 2026-09-14 — claude-code — apply same folder fix on main and push both

- Applied the identical `git mv` on `main` (commit `5f4f723`, via the main
  checkout at ~/src/HiPOZ; its unrelated local changes left untouched) and
  pushed `main` to origin. Also pushed `hipozgenai` (`fa82535..4ad406a`).
- Both branches now hold identical renames, so future merges resolve cleanly.
- Status: verified (pushes confirmed by origin refs; rename-only commits).

## 2026-09-15 — claude-code — open PR #2 to propagate hipozgenai to main

- Steve asked to propagate this branch's functionality to `main`. Verified
  `hipozgenai` (`81b29a8`) and `main` (`5f4f723`) are both in sync with origin
  and that the merge is conflict-free (`git merge-tree --write-tree main
  hipozgenai` exits 0; the intentionally duplicated data-folder renames
  `4ad406a`/`5f4f723` auto-resolve).
- Opened PR #2 (hipozgenai → main): https://github.com/vancesteven/HiPOZ/pull/2
- `gh pr merge 2 --merge` was denied by the session permission gate
  ("merge without review"), so the merge is left for Steve.
- Uncommitted working-tree changes (modified cortes2026 plot PDFs and
  `tests/test_gui_reorganization.py`; ~30 untracked files incl. root-level
  planning .md notes, `JesusCortes/`, Aug 2025 Cortes data, mahboub2026 drafts)
  are NOT part of the PR — they were never committed to the branch. Needs a
  decision on what to keep; logged in `coordination/open-questions.md`.
- Status: implemented, unverified (PR open and mergeable; propagation not yet
  observed on `origin/main`).

## 2026-09-15 — claude-code — PR #2 merged; hipozgenai propagated to main

- Steve merged PR #2 (`gh pr merge 2 --merge`; the session permission gate had
  blocked claude-code from merging it directly). Merge commit `351853f` on
  `origin/main`.
- Status: verified (`git merge-base --is-ancestor fd3718d origin/main` passes,
  so main contains the full hipozgenai branch, bookkeeping commit included).
- The local main checkout at ~/src/HiPOZ is now behind origin/main and needs a
  `git pull` there; not touched from this worktree.

## 2026-09-15 — claude-code — curate uncommitted working tree per Steve

- Steve chose per-group dispositions for the ~30 uncommitted files left out of
  PR #2. Executed: (1) `978540a` — .gitignore policy, PyQt6 fix to
  test_gui_reorganization.py, CLAUDE.md/AGENTS.md/CODEX-QUEUE/open-questions
  now tracked, four new test suites with mechanical plumbing repairs, two
  plotting docs moved under docs/, plot_all.sh + test_mccleskey_cortes.sh;
  (2) `e7b5f6f` — Aug 18-20 2025 Cortes data folders, JesusCortes/ (minus
  Office lock file), pre-20250624 benchtop CSV snapshot; (3) `af53f12` —
  mahboub2026 revision .tex drafts + generate_correct_csv.py.
- Discarded: regenerated cortes_plots PDFs (byte-identical sizes; restored
  from HEAD); deleted six root session-note .md files. Left untracked per
  Steve: Mahboub2026BenchtopData_INCORRECT.csv, mahboub2026_plots.py.save,
  precision_formatting_report.md.
- Tests run in a scratch venv (Linux container, no TeX, no libEGL):
  test_benchtop_data 5/6 (KCl reference mismatch logged in open-questions,
  NOT adjusted), test_latex_tables all pass after fixing a literal-string
  assertion, test_plot_generation 5/7 (two need LaTeX), GUI suites not
  runnable here. Status: verified for the curation itself; the test suites
  are implemented, unverified pending a macOS run.

## 2026-10-03 — claude-code — sigma(P,T) feature committed; origin/main data merged

- Committed the sigma(P,T) work held since 2026-09-15 (`1f33a65`): new
  plot_conductivity_PT in gamryPlots.py, lazy-loaded "σ(P,T)" DataSelector tab,
  tests/test_pt_map.py (8 passing headless). Plot function verified by tests
  and on the real 462-point KCl dataset; the Qt tab itself is implemented,
  unverified until launched on macOS.
- Merged `origin/main` (41b8bda) into hipozgenai (merge `58906cd`): 248
  insert-only data files — KCl std 8 S/m at low/0C, Zachary Diaz 1 bar NaCl
  0.5m, data/20260917-20260925. No code touched by the merge; no conflicts.
- Status: verified for the merge (`git merge-base --is-ancestor origin/main
  HEAD` passes; file count matches diffstat).

## 2026-10-03 — claude-code — sigma(P,T) map updated with Sept 22-25 raw data

- Fitted all 171 new raw sweeps (CPE, 10-100 kHz band) from
  data/20260922-20260925; zero fit failures. No zAnalysis configs exist for
  these dates, so sigma is provisional via the curated dataset's shared
  K_cell = 128.40 1/m (identical across all 13 curated dates, sd ~0).
- Rebuilt the sigma(P,T) map: contours from curated Aug 18 - Sep 10 data,
  new points overlaid as marked provisional triangles; companion figure of
  time-ordered sweeps shows plateau structure. Both PNGs sent to Steve.
- Finding: new sweeps step between distinct sigma plateaus (2.7-13.6 S/m) at
  fixed P,T — multiple solutions/conditions, not noise. Logged as an open
  question; NOT filtered or adjudicated here.
- Status: verified for the fits and figures (scripts in session scratchpad;
  figures delivered). PR #3 merge still pending Steve (permission gate).

## 2026-10-03 — claude-code — add plot_sigma_PT.py CLI script

- Promoted the scratchpad sigma(P,T) analysis to a committed CLI,
  plot_sigma_PT.py: curated zAnalysis loading, optional raw-date CPE fitting
  with implied/overridden K_cell, map + time-ordered sweep figures (PDF/PNG),
  provisional_sigma.csv. Tests in tests/test_plot_sigma_PT.py run against
  real repo data; 12/12 pass headless together with tests/test_pt_map.py.
- Full KCl-standard run verified; outputs in sigma_PT_plots/ (left
  untracked for Steve to commit or discard).
- Status: verified (tests + reproduced figures).

## 2026-10-03 — claude-code — 3D sigma(P,T) surface function, GUI tab, CLI output

- plot_conductivity_PT_surface in gamryPlots.py; "σ(P,T) 3D" DataSelector tab
  (interactive rotation via Qt canvas); sigma_PT_surface.pdf/.png added to
  plot_sigma_PT.py outputs (surface from curated data only, provisional raw
  points overlaid as 3D scatter). PNG from the real KCl dataset sent to Steve.
- Status: verified for the function and CLI (15/15 pt-map tests + real-data
  run); the GUI tab is implemented, unverified pending a macOS launch.

## 2026-10-03 — claude-code — conductivity isotherms (sigma vs P per T bin)

- plot_conductivity_isotherms in gamryPlots.py (1 K default bins, error bars,
  reference line); "σ(P) isotherms" DataSelector tab; sigma_P_isotherms
  output in plot_sigma_PT.py with --t-bin. 19/19 pt-map tests pass; real-data
  figure delivered to Steve.
- Observation for the pressure-trend question: several isotherms dip
  coherently near 100-125 MPa — the P ranges covered on 20260901-20260903,
  the dates with the most negative day residuals. Supports day-to-day offsets
  masquerading as pressure dependence.
- Status: verified for function and CLI; GUI tab implemented, unverified.

## 2026-10-05 — claude-code — McCleskey/WATEQ4F review for the Cortes paper

- Steve asked to confirm the McCleskey analysis accounts for neutral and
  semi-charged species via Reaktoro, matching MC12's WATEQ4F approach.
- CONFIRMED in speciation.py: neutral complexes (MgSO4(aq) etc.) are excluded
  from the conductivity sum by the charge test but constrain mass balance;
  charged complexes (NaSO4-, KSO4-, NaCO3-, HSO4-, HCO3-) map to lambda
  entries. Reference free-ion fractions re-verified INDEPENDENTLY in this
  container with USGS PHREEQC 3.8.6 compiled from source + its wateq4f.dat
  (MgSO4 free fraction 0.590 @ 0.0249 m, 0.257 @ 1.6616 m; I 0.0587/1.706;
  NaSO4- = 0.096 m in 0.3 m Na2SO4). Reaktoro itself cannot run here
  (conda-forge has no linux-aarch64 build; container is ARM) — the
  @requires_reaktoro tests skip and must be run on the Mac.
- RESOLVED the "provisional" lambda doubt: all coefficients for the charged
  complexes match the published USGS table verbatim (checked against
  www.gwb.com/data/conductivity-USGS.dat). They are effective parameters,
  valid only together with WATEQ4F speciation; comment updated.
- FOUND + FIXED: elecCondMcCleskey2012 evaluated lambda_i at per-ion I
  instead of total effective I. KCl standards 0.01/0.1/1 molal: +2.3/+4.3/
  +4.6% error before, within +-0.75% after. New anchor tests in
  tests/test_mccleskey_ionic_strength.py. Upstream PlanetProfile notified via
  ~/src/coordination/inbox/claude-planetprofile.md.
- FOUND + WIRED: the Cortes pipeline never passed speciation=True.
  cortes_mccleskey.compute_mccleskey_for_data and
  plot_cortes_with_mccleskey.py (--speciation auto|on|off) now use WATEQ4F
  speciation when Reaktoro is available. validate_speciation.py vs benchtop:
  MgSO4 RMS error 78.6% without speciation — speciation is required there.
- Status: verified for the I fix and fallback plumbing (54 tests pass, 3
  pre-existing failures unrelated); implemented, unverified for the
  speciated runtime path in the Cortes pipeline (needs Reaktoro on the Mac).
  Paper-figure regeneration awaits Steve (open-questions 2026-10-05).

## 2026-10-05 — claude-code — figure regeneration staged; GUI test mocks fixed

- Steve approved regenerating the Cortes figures. Pre-fix figures archived
  (34 PDFs) in cortes2026/figures_archive_20261005_pre_mccleskey_fix/.
- cortes2026_plots.py now passes speciation='auto' per compound;
  plot_cortes_with_mccleskey.py gains --no-tex; regenerate_cortes_figures.sh
  added — run it ON THE MAC (TeX + Reaktoro) to produce the final figures;
  this container can neither render TeX fonts nor speciate via Reaktoro.
- Smoke-verified the full cortes2026_plots.py pipeline post-wiring (22 PDFs
  into a scratch dir, fallback mode). Preview of final curves produced with
  compiled PHREEQC speciation and sent to Steve: MgSO4's old +90% model
  overshoot collapses; NaSO4- carries ~40% of sulfate in 0.5 m Na2SO4.
- Fixed MockTimeSeries in tests/test_gui_reorganization.py and
  tests/test_gui_initialization.py (missing uncertainties/colors/markers;
  AttributeError Steve hit running pytest on macOS).
- Status: verified for wiring/smoke and test fixes (26 tests pass here);
  final figures not implemented until regenerate_cortes_figures.sh runs on
  the Mac.

## 2026-10-05 — claude-code — macOS test failures triaged and fixed

- Steve's full pytest run (71 passed) surfaced: missing 'gui' fixture (7
  ERRORs) and stale GUI expectations (3-tab layout, S-column names,
  btn_create_plots, 'Ready' status, removed auto-plot handler) in the GUI
  suites; PyQt6 isSignalConnected API gap; test_plot_sigma_PT requiring
  gitignored per-user zAnalysis CSVs. All fixed; sigma_PT tests now skip
  cleanly where curated CSVs are absent and anchor paths to the repo root.
- NOT touched: test_kcl_extraction (55.88 vs 57.83) — still Steve's open
  question from 2026-09-15.
- New capability: GUI suites now run headless in the container (pip PyQt6 +
  conda-forge libegl/libgl via LD_LIBRARY_PATH, QT_QPA_PLATFORM=offscreen,
  usetex off). 48 passed / 9 Reaktoro skips across affected files. The
  sigma(P,T)/3D/isotherm tabs' construction and lazy-load wiring are now
  test-covered; on-screen visual confirmation on macOS remains the bar for
  'verified' per the UI rule.
- Status: verified (test run above; fixes exercised the changed paths).

## 2026-10-05 — claude-code — macOS suite green except open KCl question

- Steve's rerun after the test fixes: 82 passed, 9 skipped, 1 failed —
  the sole failure is test_kcl_extraction (55.88 vs 57.83 mS/cm), which
  stays red pending Steve's answer to the 2026-09-15 open question.
- The 9 skips match the Reaktoro-marked speciation tests: Reaktoro is NOT
  installed in the environment that ran pytest. Until it is
  (conda/mamba install -c conda-forge reaktoro), the 'auto' speciation mode
  falls back to total molality and regenerate_cortes_figures.sh would
  produce non-speciated MgSO4/Na2SO4 curves (it warns, but proceeds).
- Status: verified (TeX plot-export tests and both GUI suites pass on
  macOS; GUI tabs construct correctly under the real Qt platform).

## 2026-10-08 — claude-code — verify regenerated Cortes figures

- Steve reported that the regenerated figures "look the same". They are not:
  the layout is unchanged but the model curves moved. Read off
  na2so4_vs_concentration.pdf, the model values match an independent USGS
  PHREEQC 3.8.6 + wateq4f.dat speciation fed through the fixed
  elecCondMcCleskey2012 at all six (T, m) points (3.37/4.12, 3.94/4.85,
  5.09/6.35 S/m at 5/10/20 C for 0.5/0.75 m). Old values (3.59, 7.52)
  match the archive, and the fixed-I unspeciated values (3.14, 6.60) are
  ruled out. Conclusion: Reaktoro speciation engaged on the Mac, and its
  results agree with PHREEQC.
- Committed the 25 regenerated PDFs (cortes_plots + eis_plots).
- Two gaps logged in open-questions: mixtures are not speciated, and four
  pre-fix eis_plots PDFs weren't regenerated by the current script.
- Status: verified (numerical match above; figures committed).

## 2026-10-17 — claude-code — Cortes misfit investigation; units + mixture fixes; mixture speciation

- Steve reported the post-fix NaCl misfit is worse than pre-fix. Confirmed:
  NaCl RMS 1.9% (pre) -> 6.8% (Oct 8 figures). NaCl is fully dissociated in
  WATEQ4F, so only the ionic-strength fix changed it.
- Arbiter, independent of the benchtop data: IUPAC/OIML Demal KCl primary
  standards (0.1 D, 1 D at 0/18/25 C). Post-fix model is within +0.5..+2.6%;
  pre-fix was +4.4..+7.8%. The ionic-strength fix is confirmed and was NOT
  reverted. The pre-fix NaCl fit was a cancellation of errors.
- FOUND + FIXED (1): benchtop concentrations are mol/L (raw headers
  "0.5 M/L"; parser docstring says mol/L) but were stored as molality.
  New cortes2026/molarity_to_molality.py converts through PHREEQC 3.8.6 +
  pitzer.dat solution densities at 20 C (+1.1..+7.2%). Cortes2026BenchtopData.csv
  now carries true molality, with the original values in concentration_molar.
- FOUND + FIXED (2): mixture ion specs were normalized to sum 1, but the
  concentration column holds the FIRST component's concentration (notes:
  "0.5 M/L each", "1.5:0.75 M/L"). Mixture models were therefore evaluated at
  1/2 (1:1), 2/3 (2:1), 1/3 (1:2) of the true concentrations.
  get_mixture_ion_spec (moved to cortes_mccleskey.py) is now per first
  component.
- IMPLEMENTED: mixture speciation. speciation.free_ion_molalities_composition
  speciates any input composition (the single-salt path now delegates to it);
  speciated_ions_from_spec / can_speciate_spec; compute_mccleskey_model and
  resolve_speciation accept a mappable ion_spec. New Reaktoro tests are
  cross-validated against PHREEQC (0.5 m NaCl + 0.5 m MgSO4: Mg+2 0.2219;
  0.5 m Na2SO4 + 0.5 m KCl: KSO4- 0.0863).
- Results (RMS %, pre -> post; cortes2026/prepost_comparison/): KCl 2.1 ->
  2.7; NaCl 1.9 -> 5.5; Na2SO4:KCl 1:1 34.2 -> 2.3; 1:2 52.7 -> 8.1;
  NaCl:MgSO4 1:1 at 0.5-0.75 M within 2-4% (was -15..-18%).
- Remaining misfits traced to the data or the model's validity range
  (open-questions 2026-10-17): benchtop reads about +4% high vs Demal at
  1 M/20 C; Na2SO4 sigma(5 C) > sigma(10 C), which is non-physical, and
  supersaturated vs mirabilite; 1.0-1.5 M MgSO4 mixtures are far beyond MC12's
  I <= 0.7 mol/kg.
- Status: verified for the units/mixture fixes and the comparison (tests:
  tests/test_cortes_concentrations.py, 23 pass; committed comparison script
  reproduces numbers via PHREEQC). Mixture speciation via Reaktoro is
  implemented, unverified until the Mac run.

## 2026-10-17 — claude-code — fix regression: sqrt(None) when no ion has parameters

- Steve's macOS run: 103 passed, 2 skipped (the two tests that skip when
  Reaktoro is present), 2 failed. The failures were test_kcl_extraction
  (still the open reference question) and test_speciation.py::
  test_neutral_species_excluded.
- Cause: the 2026-10-05 total-I refactor initialized I as None, so a call
  whose ions all lack McCleskey parameters (the test's neutral MgSO4(aq)
  probe) hit np.sqrt(None). The old per-ion code returned 0. I now starts as
  zeros: unknown species add nothing to I and sigma stays 0. This went
  unnoticed until now because the test is Reaktoro-gated.
- Regression tests added in tests/test_mccleskey_ionic_strength.py (no
  Reaktoro needed): unknown-only input gives 0; unknown species don't change I.
- Same run verifies mixture speciation: the three PHREEQC-cross-validated
  mixture tests passed under Reaktoro.
- Status: verified (25 pass here; the failing probe now returns 0.0).

## 2026-10-17 — claude-code — NaCl "speciated values higher" query: not a bug

- Steve: speciated NaCl values vs concentration look higher than pre-fix,
  but speciation should only lower sigma. Checked against the regenerated
  nacl_vs_concentration.pdf. The post-fix MODEL is LOWER than pre-fix at
  every point (e.g. 4.363 -> 4.139 S/m at 0.5 M, 20 C). The Delta panel is
  (data - model)/model, so its increase (+2.5 -> +8.1% at 0.5 M) reflects
  the lower model. The computed values reproduce the plotted Delta exactly.
- Speciation is a no-op for NaCl: wateq4f.dat has no NaCl(aq) pair (PHREEQC
  1.546 m NaCl at 20 C gives Na+ = Cl- = 1.546). The NaCl change comes from
  the total-ionic-strength fix (down about 5%), partly offset by the molality
  conversion (data move to higher m).
- My compare_prepost_mccleskey.py used the opposite sign, (model-data)/data,
  which added to the confusion. It now uses the figures' convention
  (positive = data above model); outputs regenerated.
- Steve's Reaktoro run reproduced the PHREEQC prepost_summary.txt
  byte-for-byte (independent confirmation of mixture speciation on the Mac).
  Committed his regenerated figures.
- Status: verified (numerical reproduction of the plotted Delta values).

## 2026-10-17 — claude-code — Delta axis label states sign convention
- Per Steve: the Delta panels now read
  "Delta = (sigma_data - sigma_model)/sigma_model (%)". The label is defined
  once as plotting.DELTA_LABEL and used by plotting.py (concentration and
  temperature panels) and plot_cortes_with_mccleskey.py (both panels).
  Verified that every Delta computation (pdiff, compute_delta) is
  (data - model)/model, so the label matches the plotted values.
- Status: verified (tests/test_delta_label.py; container render without TeX
  confirms the label fits the 8x10 in figure). The Cortes paper PDFs have not
  been regenerated yet; Steve reruns ./regenerate_cortes_figures.sh on the Mac.

## 2026-10-17 — claude-code — mark McCleskey validity limit; fix mixture speciation in paper figures
- Source of the limit: McCleskey et al. (2012, GCA 77:369) validate effective
  ionic strength 0.0004-0.7 mol/kg (USGS publication page abstract).
- cortes2026/mccleskey_validity_limits.py (new) finds the first-component
  molality where PHREEQC/wateq4f I_eff = 0.7 at 5/10/20/25 C. NaCl, KCl: 0.700
  (no pairs). Na2SO4 0.291-0.297; MgSO4 0.372-0.491; Na2SO4:KCl 1:1/1:2/2:1
  0.209/0.162/0.244; NaCl:MgSO4 1:1/1:2/2:1 0.221/0.138/0.323 (5 C values).
  MCCLESKEY_LIMITS now holds the 5 C (most restrictive) values; it replaces
  the earlier unsourced table (NaCl 0.9999, MgSO4 0.01245, NH4Cl 1.034,
  Na2CO3 0.3041). NH4Cl and Na2CO3 now have no limit line (not computed; not
  in the Cortes data).
- Figures: dashed line plus light shading beyond the limit on both
  concentration panels (x-axis widened to include the line); temperature plots
  star the concentrations beyond the limit, with a footnote. Only NaCl/KCl
  0.5 m points are inside the range; all sulfate data lie beyond it.
- Bug found while wiring this: study_plots called compute_mccleskey_model
  without speciation whenever ion_spec was given, so the paper's mixture
  figures (cortes_plots/*:*) were unspeciated even though cortes2026_plots
  requested speciation. The module-level mixture speciation (Reaktoro tests,
  prepost comparison) was correct; only the figure path dropped it. Fixed in
  all three call sites. The 2026-10-17 "Speciation for mixtures" STATUS row
  is narrowed to the module; the figure path is implemented, unverified until
  Steve regenerates with Reaktoro.
- Status: limit verified (tests/test_mccleskey_validity.py, renders);
  mixture figure speciation implemented, unverified.

## 2026-10-19 — claude-code — restore Aug 18-20 2025 Cortes high-pressure data; merge Oct 2026 uploads
- Merged origin/main (a8d2c12): data/20261005-20261015 Gamry files (to about
  200 MPa), all described 'Default'; composition is not identifiable from the
  files. Logged in open-questions.
- "Missing older high-pressure plots": the Cortes plot scripts loaded only
  Aug 13-15. Aug 18 (NaCl 0.5 m, 20 C, to 598 MPa) was curated but never
  loaded; Aug 19 (-1/10/20 C) and Aug 20 (20/35/45 C, to 695 MPa) were never
  fitted (blank zAnalysis templates).
- Jesus's labelled copies (JesusCortes/Data/08.1x.25) matched to the Default_
  files by the in-file timestamp. cortes2026/curate_aug2025_hp.py sets NaCl
  0.5 m (w_ppt 28.39, as Aug 18) and excludes his flagged files (inaccurate,
  taken too early) plus six unlabelled Aug 20 files: Aug 19 15/102 excluded,
  Aug 20 57/99 (the whole 45 C sweep is flagged).
- Fitted headless with the user's --cal-map design (19,20 <- 18); borrowed
  cell constant K = 92.21 S/m-ohm (8.0 S/m standard, 11.53 ohm). The borrow
  step drops excluded rows and the exclude column from the config; the
  curation script and template backups preserve the record.
- Loader now accepts zAnalysis<folder>.csv as well as zAnalysis<date>.csv,
  and skips exclude=True rows. NaCl_0.5M_vs_pressure is coloured by T so the
  isotherms separate.
- Observations for Steve: sigma peaks at 200-250 MPa at every T; the Aug 19
  20 C descending sweep sits about 9% below Aug 18/20 at the same P (thermal
  equilibration?); Aug 18 ambient readings include Jesus's 'Interesting Test'
  and 'do not take first reading' files (spread 4.05-4.87 S/m). Not excluded
  (adjudication is Steve's).
- Status: implemented, unverified for the paper PDFs (tests/test_cortes_loader.py
  passes; container render inspected). zAnalysis configs are gitignored.

## 2026-10-19 — claude-code — read labnotes/ photos
- 10 notebook pages, 2/5/26-8/29/26 (benchtop NaCl and NaCl:MgSO4, June
  Gamry mixture runs, Aug 2026 high-pressure log). Findings logged in
  open-questions 2026-10-19 (temperature compensation before ~3/8/26;
  uncurated June Gamry mixture data with a time->composition map; Aug 2026
  leak/chiller events). No data changed. labnotes/ (40 MB JPEG) left
  untracked pending Steve's decision.
