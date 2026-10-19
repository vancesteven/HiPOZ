# Open questions — hipozgenai

Anything ambiguous, destructive, or scientifically consequential goes here and
work **stops** until Steve resolves it. Include what you were doing, what the
options are, and your recommendation.

## 2026-09-15 — uncommitted working-tree changes excluded from PR #2

While propagating hipozgenai to main (PR #2), the working tree held changes
that have never been committed, so they are not in the PR:

- Modified: 20 `cortes2026/cortes_plots/*.pdf` (regenerated plots),
  `tests/test_gui_reorganization.py`, `.gitignore`
- Untracked: root-level planning notes (`PLOTTING_ARCHITECTURE.md`,
  `CLEANUP_EXECUTED.md`, etc.), `JesusCortes/`, `data/2025081[8-20]Cortes/`,
  `cortes2026/Cortes2026BenchtopData_pre20250624.csv`, mahboub2026 drafts
  (`main_revisions2.tex`, `supplement.tex`, `generate_correct_csv.py`, plus a
  `.py.save` backup and an `..._INCORRECT.csv`), several `tests/test_*.py`,
  `plot_all.sh`, `CLAUDE.md`, `AGENTS.md`, `plans/CODEX-QUEUE.md`,
  `coordination/inbox/`, this file
- Options: (a) commit the keepers (Aug 2025 Cortes data, new tests, agent-lane
  files) and delete the junk (`.py.save`, `_INCORRECT.csv`); (b) leave all
  as-is; (c) itemized review with Steve.
- Recommendation: (c) — the mix includes lab data and manuscript drafts, which
  are scientifically consequential and not mine to triage.

## 2026-09-15 — KCl reference value mismatch in tests/test_benchtop_data.py

`test_kcl_extraction` hard-codes an expected conductivity of 55.88 mS/cm for
20C, 0.5 M KCl ("from row 8, column 7"), but `JesusData2025.csv` currently
holds 57.83 mS/cm in that cell, so the test fails (tolerance 0.1). Either the
CSV was revised after the test was written or the test's expected value is
wrong. This is a scientific-data question, so neither the expected value nor
the tolerance was changed; the test is committed as-is and fails until
resolved. The companion NaCl:MgSO4 check (expected 69.09) passes.

Steve: which number is correct for 20C, 0.5 M KCl — 55.88 or 57.83?

## 2026-10-03 — what are the Sept 22-25 "Default" runs? (affects sigma(P,T) map)

The 171 raw sweeps uploaded to main in data/20260922-20260925 (all at 1-4 MPa,
~295-296 K, description "Default") were circuit-fitted with the usual CPE
model and converted with the curated dataset's shared K_cell = 128.40 1/m.
The resulting provisional sigma values step between flat plateaus (e.g. 9/22:
5.9 -> 6.6 -> 7.9 -> 8.6 S/m; 9/25 ends 7.5 -> 9.1 -> 11.7 -> 13.6), i.e.
these are NOT one fixed 52.168-ppt standard — they look like deliberate
composition/condition steps. Only ~9% of sweeps fall near nominal 8 S/m.

Steve: what was varied in these runs (dilutions? different standards? cell
work?), and which sweeps, if any, are the 8 S/m standard? Until then they are
excluded from quantitative pressure-trend fits; the plateaus near 8.3 S/m at
1-3 MPa vs the curated 8.11 S/m at 190 MPa (same T band) would imply only a
~ -2% change over 190 MPa if they are the undiluted standard with unchanged
K_cell.

## 2026-10-05 — regenerate Cortes McCleskey figures with fixed I + speciation?

Two model corrections landed on hipozgenai (see audit, 2026-10-05):
1. elecCondMcCleskey2012 now evaluates lambda_i at the TOTAL effective ionic
   strength, as MC12 specifies (was per-ion). KCl standards: errors improve
   from +2.3..+4.6% to within +-0.75%. This shifts EVERY McCleskey curve.
2. The Cortes pipeline (cortes_mccleskey.py, plot_cortes_with_mccleskey.py)
   now supports WATEQ4F speciation ('auto': on when Reaktoro is present).
   Benchtop validation shows MgSO4 RMS error is 78.6% WITHOUT speciation.

Both change figures intended for the Cortes et al. (2026) paper. Steve:
confirm before the paper figures are regenerated with the corrected model
(run on the Mac, where Reaktoro is installed, so 'auto' speciation engages).
Also: extend speciation.VERIFIED_COMPOUNDS to Na2SO4/Na2CO3 after
validate_speciation.py's MC12+spec column checks out there — the lambda
coefficients for NaSO4-/KSO4-/NaCO3- are now verified verbatim against the
published USGS table, so the old "provisional" doubt is retired.

## 2026-10-08 — two gaps left after the Cortes figure regeneration

1. **Mixtures are not speciated.** WATEQ4F speciation is only implemented for
   single salts (speciation.SALT_RECIPES). The NaCl:MgSO4 and Na2SO4:KCl
   mixture figures therefore get the ionic-strength fix but still assume full
   dissociation. For NaCl:MgSO4 this overestimates the model, because
   MgSO4(aq) association is large (only 26-59% of Mg stays free in pure
   MgSO4 over 0.02-1.7 m). Recommendation: generalize free_ion_molalities()
   to take an arbitrary {ion: molality} input with the union of elements
   (e.g. 'Na Mg Cl S O H'). It's a small change in Reaktoro, but it changes
   the mixture figures, so it needs your OK and has to be verified on the Mac.
2. **Four eis_plots weren't rebuilt and still show the pre-fix model:**
   mccleskey_comparison/MgSO4_vs_{concentration,temperature}_mccleskey.pdf,
   mixtures/Mixtures_vs_concentration.pdf, and
   single_salts/NaCl_vs_concentration.pdf. They come from the April
   0c435df run. The current plot_cortes_with_mccleskey.py run on the default
   dates (20250813-15Cortes) doesn't produce them, apparently because those
   dates have no MgSO4 single-salt EIS data. Copies are already in the
   archive. Options: (a) regenerate them using the dates that originally
   produced them, if you know which ones; (b) remove them from eis_plots/
   so nobody mistakes them for current results (git rm; the archive keeps a
   copy); (c) leave them. Recommendation: (b), unless they are paper figures.

## 2026-10-17 — remaining Cortes model-data misfits are data or validity-range issues (Steve to adjudicate)

After the corrections (cortes2026/prepost_comparison/prepost_summary.txt):
1. **Benchtop reads high.** Cortes 1.0 M KCl at 20 C (= 1.031 mol/kg, about
   the 1 D Demal standard) measures 10.61 S/m. Demal interpolated to 20 C
   gives about 10.19 S/m, so the meter reads +4%. That accounts for most of
   the residual NaCl misfit (-1..-7%, largest at low concentration). Was the
   benchtop meter's cell constant calibrated against a standard far from
   these conductivities? Recommendation: state it in the paper, or apply a
   KCl-referenced correction factor (your call; not done).
2. **Na2SO4 benchtop data are non-physical.** sigma at 5 C exceeds sigma at
   10 C at both 0.5 and 0.75 M. 0.75 M at 5/10 C and the 2:1 Na2SO4:KCl
   mixture (1.6 m Na2SO4) are supersaturated with respect to mirabilite.
   Recommend excluding them from the model comparison, or flagging them.
3. **High-concentration MgSO4 mixtures exceed the model's validity range.**
   NaCl:MgSO4 at 1.0-1.5 M is at an effective I of about 3-5 mol/kg, vs
   McCleskey (2012)'s validated I <= 0.7. The model overshoots there by
   13-43%. Recommend marking the validity limit on the figures.
4. Supersedes the 2026-10-08 item 1 (mixtures not speciated): now
   implemented; verify on the Mac.

## 2026-10-19 — Oct 2026 high-pressure uploads: which solution is which?
data/20261005-20261015 (merged from main) are all described 'Default'. Steve
says they include KCl and NaCl high-pressure runs. Needed per date: solute,
concentration, and which files are standards, before they can be curated.

## 2026-10-19 — Aug 2025 Cortes HP data adjudication
1. Aug 19 20 C descending sweep is about 9% below Aug 18/20 at matching P.
   Keep, flag, or exclude?
2. Aug 18 'Interesting Test' (3 files) and 'do not take first reading/value'
   (2 files) are in the curated set as ordinary measurements. Exclude?
3. Aug 19-20 use the Aug 18 cell constant (--cal-map). Aug 21 has standards
   (Jesus: data inaccurate except standards). Prefer an Aug 21 calibration?

## 2026-10-19 — Findings from labnotes/ (Jesus/Rose notebook, 2/5/26-8/29/26)
1. Temperature compensation: the notebook records discovering ~3/8/26 that
   probe temperature compensation had to be turned off in settings. The 2/5/26
   room-temperature (22.4-22.8 C) NaCl readings (0.5 M: 47.02/46.92/46.75;
   1 M: 86.47/86.33/86.28 mS/cm) match 25 C literature values (about 46.8 and
   85.8), i.e. they were compensated to 25 C. Benchtop rows taken before the
   fix may be compensated: candidate cause of the ~+4% meter offset and of
   the two NaCl 20 C replicate groups (4.37 vs 4.58 S/m at 0.5 M). The CSV
   has no measurement dates; which session is each row from?
2. Probe cell factor changed 0.485 (3/17) -> 0.477 (3/19), 1.7%.
3. Not in Cortes2026BenchtopData.csv: the 1.5 M NaCl + 0.0-1.0 M MgSO4
   series (probe at RT 2/26, 10 C 3/17, 5 C 3/19; 3 readings each), and the
   Gamry runs of the same series on 6/18-6/24/26 (page 7 gives a
   time->composition map that matches data/20260618,22,24 triplet-for-
   triplet; standards 84 uS, 2764 uS, 15 mS, 80 mS at 10:21-10:38 6/18).
   Curate them? Confirm '1:2' and '2:1' are 0.75:1.5 and 1.5:0.75 M.
4. Aug 2026 KCl high-pressure log: leak at 3:30 pm 8/19 (bolts tightened);
   Lauda chiller broke 8/20; automated 30->0->30 C cycles (1 h per 5 C step)
   from 8/27 at ~76 MPa, ~86 MPa 8/28 (dropped to ~72 MPa overnight, possible
   leak), ~96 MPa 8/29. Relevant to the KCl sigma(P,T) diagnosis.
5. Pages 9, 10 and 12 were not photographed (no IMG_0203); the notes do not
   cover Aug 2025 or Oct 2026.

## 2026-10-19 — Zachary Diaz's analysis (labnotes/*.xlsx, KCl PNG): what it settles and raises
Run identities (supersedes part of "Oct 2026 uploads: which solution is which?"
and the Sep 22-25 run-identity question):
- 20260917 / 20260923 / 20260924+25: 0.5 m NaCl (28.39 ppt), 1 bar,
  T cycles at bath setpoints -1, 5, 20, 25 C (Trials 1-3; 9/23 compares
  before/after a Gamry instrument recalibration). Standard: 8 S/m, Z = 14.95 /
  14.99 / 14.83 ohm (K ~ 119-120). Sep 21-22 are not covered.
- 20261009: 5 wt% NaCl (50.021 ppt, 0.901 m), 0 -> 200 -> 0 MPa at 22.8 C;
  reversible to <0.1%. No standard row; his sigma implies K = 100.2.
- 20261014-15: the KCl-standard sigma(P) plot (23.1-23.5 C, 0-200-0 MPa)
  matches these dates. Rises 8.0 -> 8.55 S/m; one 124 MPa triplet at 8.97
  is an outlier.
- 20261005-08: still unidentified (Oct 6: two 0-200-0 sweeps; Oct 7-8: slow
  sweep).
Questions:
1. Oct 9 cell constant: 5 wt% NaCl at 20 C is 7.01 S/m (CRC), and the Cortes
   benchtop NaCl interpolates to about 7.3; his 5.945 S/m at 22.8 C is ~20%
   low. With the September K (~119.6) it would be ~7.1. Which standard set
   K = 100.2, and did the cell change between September and October?
2. His T_K is the bath setpoint; the vessel thermometer read ~300 K
   throughout the 1-bar runs. Is the setpoint the right sample temperature?
3. Trial 1 at -1 C reads 31% below his reference (1.85 vs 2.70 S/m) and the
   following 5 C step stays low: partial freezing? Trial 2 ends with three
   readings at 2.5-2.6 S/m at 25 C (Z 45-48 ohm): exclude?
4. His 'Lit Value' column (2.699, 3.134, 4.431, 4.874 S/m at -1/5/20/25 C)
   is not McCleskey; his plotted McCleskey points match our pre-fix model
   (per-ion I bug). Source of the 'Lit' table?
5. Import his tables as zAnalysis configs for these folders?

### 2026-10-19 resolution (Steve)
- Q2 above: at 0 MPa the sample holder is immersed in the PolyCool chiller
  bath; T and P at 0 MPa are the bath values. Implemented: config rows with
  P_MPa = 0 (and header P <= 5 MPa) now override the Gamry header P/T in the
  headless and GUI paths (analysis_config.bath_state).
- Q5: imported (import_zanalysis_workbook.py). Oct 9 rows carry a note about
  the unresolved cell constant (Q1). Suspect rows (Trial 1 at -1 C, end of
  Trial 2) imported unexcluded, as Zachary left them.
