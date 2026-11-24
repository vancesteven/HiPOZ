# Zachary Diaz — lab notebook transcription (pp. 13–27)

Updated: 2026-11-05. Source photos: `labnotes/IMG_0204.jpeg`, `IMG_0205.jpeg`
(pp. 11, 13; Aug 19–29) and `labnotes/IMG_0327.HEIC`, `IMG_0334`–`IMG_0340`
(pp. 14, 21–27; Aug 31–Nov 5). Values are copied as written; "R a → b"
means the |Z| range he read off the sweep (high-f plateau → 100 Hz).

## Aug 2026 — HPLT 30 °C pressure steps (p. 11, 13, 14)

- p. 11, 8/19 (heading "Jessica/Steve/Kris"): 25 °C set 2:05; 298 K /
  177 MPa at 3:05; lowered P via L1; 13 MPa, 52 MPa at 3:22, leaks seen and
  bolts tightened 3:30; 110 MPa at 3:32; 152 MPa at 3:37; T set to 15 °C.
- p. 13: 8/20 15 °C at 120 MPa (9:43) and 150 MPa (1:07); Lauda chiller
  broke; Kris automating T control. 8/27: automated T + Gamry sampling;
  P 66 → 76 MPa at 30 °C; T cycle 30 → 0 → 30 °C, 1 h per 5 °C step.
  8/28 ~86 MPa (had dropped to ~72: maybe leak). 8/29 ~96 MPa.
- p. 14: one step per day at 30 °C — 8/31 ~106 (from 90), 9/1 ~116
  (from 102), 9/2 ~126 (from 112), 9/3 ~136 (from 118), 9/4 ~146 (from 131)
  MPa. Pressure drifted down ~10–15 MPa between daily steps.

## Oct 5 2026 — 80 mS/cm KCl standard, modified 1-atm system (p. 21)

25, 20, 10, 20, 25, 20 °C, ~10 min after reaching each T. A high-frequency
"tail/kink" appeared intermittently (10 °C: 29 → 15 Ω sharp decrease; wire
re-soldered 1:38 pm). His conclusion: a few kinks do not change the
analysis; consistent bad sweeps do; removing outliers does not change values.

**This identifies data/20261005 as the 8 S/m KCl standard** at 1 atm. Its
phase-zero R steps between ~12 and ~19 Ω through the day (σ 6.1–10.1 S/m
with the Oct 14 K), matching the T steps, with six sweeps unreadable and
three at 37–360 kΩ (open circuit around the re-soldering).

## Oct 6–9 2026 — ~5 wt% NaCl under pressure (pp. 22–25)

- **Oct 6** (p. 22–23): load sample 1:59 pm; up 50 → 200 → 76 MPa with
  ~10 min per step, then 1 atm 4:15, and a second cycle 25 → 200 → 75 MPa
  → 1 atm 5:58 pm (~5–10 min steps). Valve sequence L1/H1/H2/H4 recorded.
  "Discrepancies between going up vs down … suspect temperature
  equilibration, but going down yielded higher conductivity … may be a
  connectivity issue." → data/20261006 (90 sweeps).
- **Oct 7–8** (p. 23–24): repeat with **≥ 30 min** per step: 1 atm 10:05 →
  25 … 200 (2:54 pm) → 100 MPa (5:14 pm); left overnight, read 58 MPa at
  9:21 am Oct 8 (dropped from 100 to 75 then 58), 25 MPa, 1 atm 10:33.
  Room T ~295 K vs 296 K the day before. → data/20261007, 20261008.
  Conclusions: 30 min is best for T equilibration; 10 min reproduces the
  descending-then-ascending trend of the initial HPLT data; 5 min gives a
  large up/down discrepancy and scatter within triplets.
- **Oct 9** (p. 25): repeat with 10 min: 1 atm 11:02 → 200 MPa 12:55 →
  1 atm 2:31 pm (overshot 25 MPa: 0 → 30 → 20 → 25). "Showed the ascending
  trend, consistent with the 30-min run … conflicts with the initial HPLT
  data with the ↓↑ trend / mid-dip. Will repeat with the standard."
  → data/20261009.

## Oct 14–15 2026 — 80 mS/cm KCl standard under pressure, room T (p. 26–27)

Swapped the 5 wt% NaCl for the KCl standard. 1 atm 9:41 (18.7 → 575 Ω),
25, 50 MPa; **pressure leak at 11:39** (75 → 70 MPa in 30 min; oil leak at
the pump hand, retightened; some oil leaked into the bottom bottle); 75 MPa
at 12:11 after more tightening; 100 … 200 (3:21 pm) → 125 MPa (5:01 pm);
overnight 125 → 98 MPa; Oct 15 100, 75, 50, 25 MPa, 1 atm 12:58.
Conclusion (p. 27): no ↓↑ trend or mid-dip, consistent with the 5 wt% NaCl
trial; "suggests a potential error with the measurements in the initial
HPLT data. The current setup occasionally produces outliers still."

## Nov 5 2026 — standard at 25 °C, 0–200 MPa, new chiller (p. 27)

1 atm 11:35, then 25 MPa steps to 200 (2:11 pm) and back to 1 atm 4:28 pm;
20 min per step to 100 MPa, 10 min above. → data/20261105.

## What this settles

| Open item | Answer from the notebook |
|---|---|
| Oct 5–8 identity | Oct 5: 80 mS KCl standard, 1 atm, T steps. Oct 6–8: ~5 wt% NaCl σ(P), 10-min (Oct 6) and 30-min (Oct 7–8) steps |
| Oct 14 124 MPa / Oct 15 52–100 MPa poor fits | fit jumps (refit_chained.py); the Oct 14 leak and tightening happened 11:39–12:11, just before the 75–124 MPa steps |
| Oct 6 up/down discrepancy | his diagnosis: too short an equilibration (5–10 min); 30 min removes it |

Still open: the Oct 9 cell constant. His Oct 9 workbook implies K = 100.2.
With the Oct 14 standard's K (120.93, phase-zero basis), the Oct 6–9 NaCl
reads ~7.3–7.9 S/m at 1 atm, 296 K, much closer to the ~7.0–7.3 S/m
expected for 5 wt% NaCl than his 5.945. That K was measured five days later
and after the leak repair, so it is an estimate, not a calibration.

---

# Pages 28–37, Nov 6–23 2026

Pages 28–37 (labnotes/IMG_6027–IMG_6036.heic), transcribed 2026-11-23 by a Claude agent from the page images; not yet checked by the authors.

Page numbers: p. 28 (IMG_6027), p. 32 (IMG_6031), p. 35 (IMG_6034) and p. 36 (IMG_6035) are printed on the images; the others follow the sequence (IMG_6028 = 29, 6029 = 30, 6030 = 31, 6032 = 33, 6033 = 34, 6036 = 37) and are not visible in the photos. Each step line is "clock time - pressure". Temperatures are the nominal setpoints Zach wrote; the logged vessel T in the data files runs about 1 °C above them (294 K for "20 °C", 289 for 15, 284 for 10, 279 for 5, 303 for 30). "[?]" = unreadable; "[word?]" = uncertain.

## Nov 6 2026 — standard at 20 °C, 0–200 MPa (p. 28)

Step list (pm): 12:40 1 atm; 12:55 25; 1:08 50; 1:22 75; 1:36 100; 1:51 125; 2:12 150; 2:28 175; 2:48 200; 3:02 175; 3:16 150; 3:31 125; 3:45 100; 4:02 75; 4:19 50; 4:33 25; 4:48 1 atm. Beside the first step: "- wait 10 min". Margin note:
"* impedance looks generally higher compared to yesterday (11/5/26) at 25°C
 - adjusted Chassis Ground before 100 MPa measurement
 - no issue in GUI"

## Nov 9 2026 — standard at 15 °C, 0–200 MPa (p. 29)

12:35 pm 1 atm; 12:48 25; 1:02 50 (50 written over another number); 1:16 75; 1:34 100; 1:48 125; 2:03 150; 2:17 175; 2:33 200; 2:46 175; 3:00 150; 3:13 125; 3:27 100; 3:40 75; 3:56 50; 4:07 25; 4:23 1 atm. No comments.

## Nov 10 2026 — standard at 10 °C, 0–200 MPa (p. 30)

12:08 pm 1 atm; 12:25 25; 12:39 50; 12:52 75; 1:06 100; 1:19 125; 1:33 150; 1:47 175; 2:00 200; 2:13 175; 2:27 150; 2:40 125; 2:53 100; 3:08 75; 3:22 50; 3:36 25; 3:48 1 atm. Note: "* Gamry wires were moved to new power strip closer to the wall." (No time given.)

## Nov 11–12 2026 — standard at 5 °C and 30 °C, 0–200 MPa (p. 31)

Left column, "11/11/26 standard at 5°C, 0-200 MPa": 9:10 am 1 atm; 9:25 25; 9:39 50; 9:52 75; 10:07 100; 10:20 125; 10:34 150; 10:52 175; 11:07 200; 11:20 175; 11:35 150; 11:50 125; 12:04 pm 100; 12:18 75; 12:31 50; 12:44 25; 12:58 1 atm.
Right column, "& 30°C, 0-200 MPa": 3:27 pm 1 atm; 3:42 25; 3:56 50; 4:09 75; 4:23 100. Circled "11/12/26", then: 9:26 am 125; 9:39 150; 9:52 175; 10:08 200; 10:21 175; 10:34 150; 10:47 125; 11:01 100; 11:15 75; 11:30 50; 11:44 25; 12:00 pm 1 atm.
Below: "4:20 pm - swapped out KCl standard w/ 5 wt% NaCl" then, underlined, "wire broke" and "- remade connection". (The text is under the Nov 12 column and carries no date of its own; whether the break was found before or after the swap is not stated.)

## Nov 16 2026 — 5 wt% NaCl at 30 °C, 0–200 MPa (p. 32)

First attempt: 9:50 am 1 atm; 10:09 50; 10:33 100; 10:48 150; 11:04 200; 11:24 150; side note "waited 10 min," with a long down-arrow to "needs more time to equilibrate".
"Redo. 15/20 min. wait" (as written; the first number is overwritten, likely 15 over 20 [?]): 3:00 pm 1 atm "- 20 min."; 3:25 50 (down-arrow); 3:48 100 (↓); 4:01 150 (↓); 4:21 200 ("-"); 4:46 150 "- works w/ 15 min."; 5:06 100; 5:25 50; 5:45 1 atm (down-arrow beside the last four).

## Nov 17–18 2026 — 5 wt% NaCl at 25, 20, 15 °C (pp. 33–34)

- "11/17/26, 5 wt% NaCl at 25°C, 0-200 MPa  - wait 15 min.": 9:45 am 1 atm (the leading digit looks like a 4 or 9 [?]; 9:45 is read from the sequence and confirmed by the data); 10:07 50; 10:25 100; 10:43 150; 11:03 200; 11:22 150 (written over); 11:44 100; 12:02 pm 50; 12:21 1 atm.
- "5 wt% NaCl at 20°C, 0-200 MPa": 1:35 pm 1 atm; 1:56 50; 2:15 100; 2:34 150; 2:54 200; 3:16 150; 3:38 100; 3:55 50; 4:13 1 atm.
- "5 wt% NaCl at 15°C, 0-200 MPa": 4:52 pm 1 atm; 5:19 50 [word struck out] MPa; 5:36 100 MPa. Continued p. 34, circled "11/18/26": 9:27 am 150; 9:46 200; 10:06 150; 10:25 100; 10:46 50; 11:03 1 atm; 11:12 1 atm.
- Note with an arrow to the Nov 17 4:52 pm 1-atm line, "(4:52pm 11/17/26)": "-1st 1 atm measurement is off due to not waiting long enough for temp. to settle. Wait 1 hour btw 5°C minimum" and "* may vary".

## Nov 18–19 2026 — 5 wt% NaCl at 10, 5, ~0 °C (pp. 34–35)

- "5 wt% NaCl at 10°C, 0-200 MPa" (Nov 18): 12:30 pm 1 atm; 12:50 50; 1:10 100; 1:28 150; 1:47 200; 2:05 150; 2:25 100; 2:44 50; 3:04 1 atm.
- "5 wt% NaCl at 5°C, 0-200 MPa": 4:05 pm 1 atm; 4:25 50; 4:45 [struck-out time] 100; 5:03 150; 5:22 200; 5:41 150; 6:00 100; circled "11/19/26": 9:27 am 50; 9:46 am 1 atm.
- "11/19/26, 5 wt% NaCl at ~0°C, 0-200 MPa" (p. 35): 11:17 am 1 atm; 11:37 50; 11:56 100; 12:27 pm 150; 12:47 200; 2:10 150; 2:29 100; 2:49 50; 3:08 1 atm. Side note: "* chiller set to -1°C / -Chamber reading is warmer by 2°C."

## Nov 19 2026 — calibration in an external chiller (p. 35)

"Moved sample out to external chiller to calibrate standard.
Took meas. [3:52 struck out] 3:53 pm. - Impedance graph is off. -> Compare to data to see if moving the sample holder changed reading / calibration.
[struck out: 2nd attempt at 4:0? pm.]
Got it to work at 4:45 pm = 5. wt % NaCl σ ≈ 1.5 S/m [value as written; see below]
 -needed angle/put tension on wires in a specific way that controlled impedance.
True Standard at 5:21 pm. 8 S/m  (double-underlined)
 ~deleted other calibration tests
 -Swapped in [struck out] 10 wt % NaCl and confirmed correct [struck out] impedance [struck out] curve -5:51 pm"

## Nov 20 2026 — 10 wt% NaCl (1.9 m?) at ~0 and ~5 °C (p. 36)

- "11/20/26, 10 wt% NaCl at ~0°C, 0-200 MPa (1.9 m?)": 9:15 am 1 atm; 9:34 50; 9:51 100; 10:11 150; 10:29 200; 10:50 150; 11:14 100; 11:57 50; 12:17 pm 1 atm. Beside the 150 MPa (10:50) line: "X2 - then again at 11:37 am, ↑ P direction" (meaning not clear from the text alone: the 150/100 steps appear to have been taken twice, a repeat starting 11:37 am on the way up).
- "10 wt% NaCl at ~5°C, 0-200 MPa": 1:35 pm 1 atm; 1:55 50; 2:14 100; 2:33 150; 2:51 200; 3:10 150; 3:28 100; 3:45 50; 4:05 1 atm.

## Nov 20–23 2026 — 10 wt% NaCl at ~10 and ~15 °C (p. 37)

- "10 % wt NaCl at ~10°C, 0-200 MPa" (Nov 20): 5:00 pm 1 atm; 5:19 50; 5:38 100; 5:57 150; 6:15 200; 6:33 150.
- "11/23/26": [9?]:38 am 100 MPa; 9:56 50 MPa; 10:15 1 atm (the leading digit of the first time is unreadable, 9 is inferred from the sequence). The Nov 20 descent from 150 MPa thus stopped overnight and was finished on Nov 23, and nothing is written for the 100 MPa step on Nov 20 evening.
- "10% wt NaCl at ~15°C, 0-200 MPa" (Nov 23): 11:15 am 1 atm; 11:34 50; 11:53 100; 12:12 pm 150; 12:30 200; 12:48 150; 1:07 100; 1:25 50; 1:43 1 atm.

## What pp. 28–37 settle or raise

| Item | What the notebook says | Consequence for the data |
|---|---|---|
| Equilibration per step | Nov 6: wait 10 min. Nov 9–12 steps are ~13–15 min apart, 20 min on Nov 10 (Nov 5: 20/10 min). Nov 16 first run: "waited 10 min … needs more time to equilibrate"; redo with 20 min ("works w/ 15 min"). Nov 17: "wait 15 min". Nov 17–19 steps are ~18–20 min apart | Nov 16 9:50–11:24 am run is not in data/20261116 (files start 15:00): consistent with its being discarded. Nov 16 pm 15–20 min per step |
| Nov 6 chassis ground | "adjusted Chassis Ground before 100 MPa measurement", also "impedance looks generally higher compared to yesterday (11/5/26) at 25°C", "no issue in GUI" | The adjustment falls at the 1:36 pm step: sweeps 20261106_133700, 133754, 133848 (100 MPa) and every later sweep of the day; the 1 atm–75 MPa sweeps (124105–132420) are before the change. A step between 75 and 100 MPa in Nov 6 R could be this and not pressure |
| Nov 10 wiring | "Gamry wires were moved to new power strip closer to the wall" (no time) | Nov 10 sweeps 120757–155042 may or may not all be after the move; compare Nov 9 vs Nov 10 R at 1 atm |
| Nov 12 76 MPa (±12 Ω fits) and 52 MPa | Nov 12 steps 75 MPa (11:15) and 50 MPa (11:30) carry no comment; no leak or problem is recorded for Nov 12 | The notebook does not explain the ±12 Ω fits or the 13.30 Ω at 52 MPa. It neither supports nor argues against the provisional exclusion |
| Soldered-connection break and repair | p. 31, below the Nov 12 columns: "4:20 pm - swapped out KCl standard w/ 5 wt% NaCl", "wire broke", "- remade connection". No time or date for the break itself. No KCl standard was run after it before the NaCl series (Nov 16 starts with the 5 wt% NaCl; no entry for Nov 13–15) | The break and repair lie between the last pre-repair KCl sweep (Nov 12 11:59 1 atm) and Nov 16 15:00. K = 120.6 (Nov 5–12) applies to Nov 5–12 only; the 4% change across the repair is consistent with a remade connection |
| Post-repair calibration, Nov 19 | The "True Standard at 5:21 pm. 8 S/m" is the first standard after the repair; it was taken in an external chiller (sample moved out, "to calibrate standard") after the 3:08 pm 1 atm of the ~0 °C run. The 3:53 pm measurement was "off", the 4:0? pm second attempt was struck out, 4:45 pm "got it to work" and "needed angle/put tension on wires in a specific way that controlled impedance". "~deleted other calibration tests" | In the data: 164412–164602 (T 287 K) are the 4:45 pm attempt (R 15.5 Ω); 171146–171334 are unlisted in the notebook (R 14.55 Ω); 172145–172334 (R 14.57 Ω, T 297.5 K) are the "True Standard at 5:21 pm". No sweeps exist at 3:53 or 4:0x pm: these are the "deleted other calibration tests". So the standard run selected for K (Nov 19 17:21) is the one Zach marks as true |
| Wire orientation | Wire angle and tension "controlled impedance"; the 5 wt% NaCl at 4:45 pm was written "σ ≈ 1.5 S/m", about a fifth of the ~6–7 S/m expected for 5 wt% NaCl; the reading is far below what 5 wt% NaCl gives in the vessel (~6 S/m at 16 °C), so it is probably a provisional value from the unsettled external setup; not resolved | Supports "bad wire orientation" as a reason for the poor early Nov 19 external readings |
| Nov 19 17:50 sweeps | The notebook at 5:51 pm: "Swapped in 10 wt % NaCl and confirmed correct impedance curve". The 17:50 sweeps (R 10.1–10.4 Ω, 285 K, 175006–175155) match this time. 115.2 m⁻¹ / 10.1 Ω is 11.4 S/m, in the range of 10 wt% NaCl near 12 °C | These are likely a 10 wt% NaCl check, not a KCl calibration with bad wire orientation as the pipeline labels them (diaz_hp_sweep_fits.csv). The exclusion from the 5 wt% isotherms stands, but the reason and run label should change: see Proposed actions |
| Nov 17 first 1-bar at 15 °C | "(4:52pm 11/17/26) 1st 1 atm measurement is off due to not waiting long enough for temp. to settle. Wait 1 hour btw 5°C minimum" | The 16.2 °C isotherm's first ascending 1-bar sweeps (20261117_165100, 165155, 165250) are flagged by Zach as off; R is 18.77 Ω there against 18.97–18.99 Ω at 11:03–11:12 Nov 18 (1 atm, same T). The main figure anchors each isotherm on its first ascending 1-bar step, so the 16.2 °C anchor is built on a step the author calls unsettled |
| 1-bar offset that grows on cooling | Step spacing is ~20 min per step on every isotherm, and the first 1-bar step of an isotherm comes 39 min (15 °C: 4:52 pm after 4:13 pm), 78 min (11.5 °C: 12:30 after 11:12 am), 61 min (7 °C: 4:05 pm after the 3:04 pm 1 atm) and 91 min (2 °C: 11:17 am after 9:46 am the previous day, ~0 °C setpoint) after the last step of the previous temperature, i.e. only the 15 °C step is below the author's rule "Wait 1 hour btw 5°C minimum" ("may vary"), and Zach himself flagged that one as off. Nov 19: "chiller set to -1°C / Chamber reading is warmer by 2°C". Zach did not compare the NaCl against McCleskey | The notebook offers thermal lag as a documented risk (it was already seen at 15 °C), supporting the thermal-lag branch of the open question; it records nothing about a series resistance or K change besides the repair |
| Vessel T versus setpoint | Data T is ~1 °C above the nominal on every day (e.g. nominal 20 °C = 294 K); the isotherm labels in the figures use the data (24.8, 20.5, ...) | No action; the chamber-vs-chiller offset is noted on Nov 19 only |
| Leaks | No leak is recorded on pp. 28–37 | none |
| Nov 20–23 | 10 wt% NaCl at ~0, 5, 10, 15 °C. No data/20261120–20261123 folders exist in the repo | Not yet transferred |

### Timestamp cross-check (notebook vs data/202611*/ConductivityData_Default)

Every dated step list on pp. 28–35 has a sweep triplet within 1–2 min of the notebook time (sweep timestamp is the first of three; offsets are up to ±1.8 min, the largest being Nov 17 9:45 am vs 9:46:47). Exceptions:
- Nov 16 9:50–11:24 am first run: no sweep files (data/20261116 starts 15:00:09).
- Nov 16 4:01 pm 150 MPa and 4:21 pm 200 MPa: nearest sweeps 16:07:43 and 16:26:43 (+6.7 and +5.7 min; every other Nov 16 step is within 1 min).
- Nov 19 3:53 pm and 4:0x pm external-chiller attempts: no files (last file 15:08, next 16:44). 4:45 pm = 164412; 5:21 pm = 172145; 5:51 pm ≈ 175006.
- Nov 19 17:11 (171146–171334): sweeps in the data with no notebook line.
- Nov 18 11:03 and 11:12 am (two 1 atm entries): both present as 110317 and 111211.
- Nov 20, Nov 23 (pp. 36–37): no data folders; cannot be checked.
- No sweep in the data lacks a notebook time apart from Nov 19 171146–171334.

### Proposed actions (for Steve) — not applied

1. data/20261119 sweeps 20261119_175006, 175101, 175155: relabel in diaz_hp_sweep_fits.csv from "out-of-vessel KCl calibration … suspected bad wire orientation" to "10 wt% NaCl check, 5:51 pm (Zach: 'confirmed correct impedance curve')", still excluded from the 5 wt% isotherms and from K. Confirm with Zach whether the 10 wt% sample was in the vessel or the external chiller.
2. 20261117_165100, 165155, 165250 (first 1-bar step of the 16.2 °C isotherm; Zach: "1st 1 atm measurement is off due to not waiting long enough for temp. to settle"): consider anchoring the 16.2 °C isotherm in the main figure on the 1-bar step at the end of the isotherm or on 20261118_110317 / 111211 (11:03 and 11:12 am 1 atm) instead of the first ascending step. The other cold isotherms' first 1-bar steps (20261118_123101, 160528, 20261119_111751) were 61–91 min after the previous temperature step, so they meet his 1 h rule; the 1-bar offset on cooling is therefore not explained by this note.
3. 20261106_133700 onward (100 MPa and every later Nov 6 sweep): record "chassis ground adjusted before 100 MPa" in the config comment; do not exclude without comparing R across 75 → 100 MPa.
4. Nov 12 76 MPa / 52 MPa: the notebook gives no reason to keep or drop the provisional exclusion; Steve to decide.
5. Ask Zach: date and time of the wire break; why there is no KCl standard between Nov 12 and Nov 19; what "σ ≈ 1.5 S/m" on Nov 19 4:45 pm refers to; whether the 11:37 am repeat on Nov 20 is a second ascent.
6. Do not apply K = 120.6 (pre-repair) to Nov 16–23; K applies from the Nov 19 17:21 standard (115.2) or the self-calibration already adopted.
