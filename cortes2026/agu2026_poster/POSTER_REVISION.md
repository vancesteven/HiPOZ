# Poster revision 1 (2026AGU_Vance_Electrical -> 2026AGU_Vance_Electrical_rev1.pptx)

Built by `build_poster_rev1.py` (original opened read-only; md5 of original unchanged, 77b7b72f...). Shapes matched by name + text.
Slide 44 x 44 in, 1 slide. Shape count 75 -> 79 (+4 new band shapes, +1 Fig C, -1 old picture).
Positions in inches (x, y, w, h). Status: implemented, unverified in PowerPoint (schematic preview only; see end).

## Added
| shape | position | content |
|---|---|---|
| Rev1 band heading | 0.5, 31.9, 25.2 x 1.4 | 40 pt bold "New: pressure raises conductivity more in colder oceans"; 28 pt "NaCl 0.5 mol/kg to 700 MPa (0.8-35 °C); NaCl 0.9 mol/kg (1.7-29 °C) and KCl 0.70 mol/kg (6-30 °C) to 200 MPa". Theme font (as idx "TextBox 134", no typeface set). |
| Rev1 Fig A sigma vs P | 0.5, 33.6, 17.3 x 6.49 | fig_A_sigma_P.png |
| Rev1 Fig B relative sigma | 18.0, 33.6, 7.79 x 6.49 (ends x 25.79) | fig_B_relative_sigma_0p9m.png |
| Rev1 band method text | 0.5, 40.3, 25.3 x 1.7 (ends y 42.0) | 20 pt, 2 paragraphs (Method; Preliminary), see build script |

## Changed
- "Straight Connector 117" (green divider): was x 25.96, y 20.30, h 21.02 (to y 41.32); now h 11.30 (ends y 31.6). Other lines untouched.
- "Picture 150" (38.45, 33.09, 5.82 x 4.31, ran to x 44.27) deleted; replaced by "Rev1 Fig C pct increase vs T" at 38.75, 33.25, 3.70 x 3.17 (same z-order slot). Moved 0.3 right / 0.16 down from the old top-left so it clears the Adams & Hall scan (right edge x 38.67) and the 40 pt section heading box (bottom y 33.21). Width limited to 3.7 in because the text below must fit at >= 18 pt (see Text fit); Fig C's own text is therefore ~11-13 pt at print size.
- "TextBox 151" (was 39.02, 37.37, 4.68 x 4.34, auto-fit, default 18 pt):
  now 38.75, 36.52, 4.95 x 5.48 (ends y 42.0), fixed size, no auto-fit, explicit 18 pt, 4 paragraphs with 4 pt space after (blank spacer paragraphs removed).
  Before: "New measurements in 5wt% NaCl (aq; ~1 molal) show previously published increasing conductivity with pressure to 200 MPa. The change is nearly 10% (equivalent to about 1.5°C temperature change). / This behavior contradicts extrapolations from above 200MPa by Pan et al. (2021) / [Wingdings arrow] Magnetic induction studies ... relevant solutions".
  After: the four paragraphs in the build script (Wingdings arrow replaced by plain "→"; the italic on "increasing" is gone).
- "TextBox 114" (footnote): was "*Thermo/Orion Star A329 Image - Please add the following disclaimer: Reference herein ... / /" ; now "*Reference herein to any specific commercial product, ... California Institute of Technology." Same 12 pt Roboto-Light run formatting; trailing line breaks removed; box height 0.91 -> 0.55 (position 27.82, 31.67 unchanged).

## Numbers and sources (poster_facts.md)
- 0.5 m NaCl range 0.8-35 °C, to 700 MPa: per-isotherm table (T 0.85-34.85; highest P 599-690 MPa).
- 0.9 m NaCl 1.7-29 °C: "11.2% at 1.7 C ... 5.8% at 29.3 C" (section a). KCl 6-30 °C: "12.7% at 6.3 C ... 7.1% at 29.5 C".
  CORRECTION vs brief: brief said "(2-30 °C)" for both; the facts give 1.7-29.3 (NaCl 0.9) and 6.3-29.5 (KCl), so each is stated separately.
- 6% at 25-29 °C: 6.23 (24.9), 5.84 (29.3), 6.64 (23.0). 11% at 2-7 °C: 11.22 (1.7), 10.62 (7.0). KCl 7-13%: 7.09-12.71.
- 3-4 °C equivalent: section b, dT 3.1 K at 24.8 C, 3.8 K at 1.7 C, 4.0 K at 7.0 C (MC12).
- 200-250 MPa maximum (0.5 m): section "0.5 mol/kg NaCl only": highest-sigma step 198-250 MPa (parabola 206-259, guide 219-277). 200-250 follows the step/parabola values; the guide polynomial gives up to 277 (author to decide).
- Pan +1% to -10%: section d, "-10.4% to +1.0%" (0.9 m).
- Pan fit data 212-1713 MPa: section d validity.
- 13-22% (0.5 m, Aug 2025): 12.9-21.9 excluding the 9.85 C isotherm (31.5%, flagged "1-bar step unsettled?"). About twice: "about 2.1x" (section h). Adams & Hall +6.6% at 25 °C: section c.
- Up/down agreement: CORRECTION. Brief said 0.3 pp. Table: Nov 2026 runs 0.19-0.34 pp (NaCl 0.9 Nov: 0.20-0.32; KCl: 0.19-0.34); Oct 7-8 NaCl run is 0.80 (not "Nov"). Poster states "within 0.34 percentage points (20-190 MPa; ...)": the facts define up/down over 20-190 MPa, not "at 200 MPa".
- Anchoring to McCleskey (2011): section g (McCleskey et al. 2012 is used below 5 °C; the poster text says only 2011).

## Missing from the reference footer (not edited)
Cited on the poster (text, labels or figures) but absent from the reference list: Mahboub et al. 2026 (label near the QR code), McDougall 2011 (Gibbs Seawater, in the induction text box), Chang et al. in prep, McCleskey et al. (2012) (Fig A/B/C footnotes "McCleskey (2011, 2012)" and the poster's anchoring below 5 °C). Present in the list: Adams & Hall 1931, McCleskey 2011, Pan 2021, Psarakis 2024, Larionov & Kryukov 1984, Vance 2018/2021, Zolotov & Kargel 2009, Petricca 2026, Guo & Keppler, Sinmyo & Keppler, Kraichman.

## Deliberately untouched (author decisions)
Title, authors, affiliations, left/middle modelling content, benchtop section, references, QR codes, "This can run on your phone", empty placeholders, stray "*" (TextBox 109), the literature figures and labels (compilation, Adams & Hall scan, Larionov, labels), the 5wt%/"nearly 10%" framing elsewhere.

## Open items
- Text fit of the pressure box is an estimate (see below). Real PowerPoint rendering (Aptos) must be checked by the author.
- The footnote's "*" marker has no matching asterisk next to the Orion photo except the stray "*" at (32.8, 14.2).
- Idx "Straight Connector 117" now stops at y 31.6; the footnote (y 31.67-32.2) and the heading "How Does Pressure..." sit right of it.
