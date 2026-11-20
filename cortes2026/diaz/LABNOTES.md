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
