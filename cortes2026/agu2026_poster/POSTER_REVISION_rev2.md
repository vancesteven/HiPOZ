# Poster revision 2 (2026AGU_Vance_Electrical.pptx -> 2026AGU_Vance_Electrical_rev2.pptx)

Built by `build_poster_rev2.py` (re-runnable; `--title "..."` sets the title; `--no-preview` skips the PNGs). The ORIGINAL draft is
opened read-only; md5 77b7b72fc411c5a67754c287bb0c6763 before and after (verified in the script output). `2026AGU_Vance_Electrical_rev1.pptx`
untouched (md5 3681b516...). Shapes are matched by name + text + position, never by index alone.
Status: **implemented, unverified in PowerPoint**. Verified artifacts: script output (1 slide, 44 x 44 in, no shape outside 0-44 or inside the 0.4 in margin,
overlap list, text-fit table) and the schematic previews `poster_rev2_preview.png` (3300 px), `poster_rev2_top.png`, `poster_rev2_middle.png`,
`poster_rev2_bottom.png` (6600 px wide each).
**The previews use DejaVu Sans, a substitute for the theme font Aptos (about 10-15 % wider). Superscripts are drawn as plain digits. EMF figures (induction response, circuit) are drawn from the bitmaps extracted earlier. Real PowerPoint rendering (title size, text boxes, EMF figures) still has to be checked by the author.**

## Final grid (inches from top-left, x0-x1 / y0-y1)
| region | position | content |
|---|---|---|
| Header | y 0.5-5.62 | title box 0.5-31.5 (88 pt bold, 2-line height reserved; `TITLE` constant, shrinks stepwise to 66 pt); authors 28 pt bold (2 lines); affiliations 18 pt; logo 33.39-43.5 x 0.9-3.22 |
| Row 1 left | x 0.5-13.5, y 5.92-21.88 | heading 40 pt; body 24 pt (y 6.92-9.86); induction figure 12.0 x 9.09 at (1.0, 10.01); explanatory text 24 pt y 19.21-21.88 |
| Row 1 hero | x 14.0-26.0, y 5.92-21.07 | heading 40 pt (2 lines); Fig B 12.0 x 10.0 at y 7.71; takeaway box 26 pt y 17.91-21.07 |
| Right column, benchtop | x 26.48-43.6, y 5.92-26.08 | heading 32 pt (y 5.92-7.29), QR/photo/table/circuit/plots as a block (dx -0.16, dy -4.58 vs draft) |
| Right column, earlier work | x 26.48-43.6, y 26.78-39.13 | 32 pt sub-heading, 24 pt subtitle, literature figures at 125 % |
| Row 2 | x 0.5-26.0, y 22.42-33.5 | heading 40 pt + 26 pt line (22.42-23.83); Fig A 25.5 x 9.56 at y 23.93 |
| Row 3 | x 0.5-26.0, y 34.05-40.65 | Fig C 7.0 x 6.0 + 22 pt caption (0.5-8.1); What we find 8.5-19.7 (24 pt); Method 20.1-26.0 (22 pt) |
| Footer | x 0.5-43.5, y 41.55-43.55 | references, 18 pt (the draft's default size; it fits, so not reduced) |
| Separators | gray 2.5 pt straight lines | header (y 5.67), x 13.75, x 26.24, y 22.14, y 33.77, y 41.35; old green lines deleted |

## Deviations from the brief (and why)
1. **"Earlier work" block is in the right column under the benchtop block (y 26.8-39.1), not in Row 3.** With the induction figure gone, the benchtop block
   ends at y 26.1; a Row 3 with four blocks would have left a 17 x 8 in hole under it. Row 3 therefore has three blocks (Fig C, What we find, Method) across x 0.5-26.0.
2. **Earlier-work block scaled UP to 125 %** (brief allowed only down to 75 %) so it fills the column width; label font sizes in it were scaled with it
   (14 -> 18 pt; 15 -> 19 pt). The Adams & Hall scan, the Larionov scan and their labels were shifted 0.75 in (draft units) to the right of the compilation plot because in the draft they overlapped
   the compilation's right edge (the draft overlap was real: content of the compilation plot reaches 98 % of its width). Relative positions within the block are otherwise preserved.
3. **Benchtop block is rigid except for the heading**: the draft's gap between the heading and the QR code was filled by the induction figure; after removal the rest of the block was moved up by
   dy = -4.58 (relative to the draft) so that no 6 x 17 in hole remains. Small non-rigid fixes inside it (draft coordinates): TextBox 104 y 18.32, TextBox 106 y 18.74, TextBox 98 y 20.29, circuit image x 32.12 / y 23.36 (was overlapping the melting-line diagram), TextBox 105 y 24.34,
   box heights of those boxes cut to the text; the meter photo's blank bottom strip (5.7 %) cropped (non-destructive crop). Text sizes in the benchtop boxes are unchanged (14 pt for "Measurements were conducted by ...", the draft's size; below the 20 pt body minimum, kept as the brief says to keep sizes).
4. **Disclaimer footnote is beside (left of) the meter photo, not under it**: directly under the photo are the "Benchtop conductivity sensor" text boxes. A "*" box (Rev2 photo asterisk, 32 pt) sits between footnote and photo. Footnote is 14 pt (draft 12 pt), which is allowed to be below 20 pt.
5. **Induction figure 12.0 in wide, not 13.0**: at 13.0 in the column would run to y about 22.9 (brief: Row 1 ends at 21.5; I allowed 21.9).
6. Header: all author names are bold (in the draft only the first run was bold and the superscripts were plain); numbers superscripted in authors and affiliations (draft superscripted most, not the first affiliation number); authors/affiliations left-aligned (draft affiliations were centered). The title box height reserves two lines at 88 pt. **A 140-character title cannot fit on two lines at 80-88 pt in a 31 in box** (about 84-100 characters is the limit); the script steps down to 66 pt for two lines, and if that is still too long it uses 3 lines at 66 pt and prints a WARNING (header and Row 1 would then be tight).
7. Heading "How Does Pressure Affect Electrical Conductivity?": 32 pt bold (draft: 40 pt underlined, not bold), made to match the other bold headings.
8. Preview: the brief's schematic preview renderer was reused only in spirit; the preview code is inside `build_poster_rev2.py` and reads the EMF bitmaps from the recon folder (env `POSTER_RECON`).

## Shapes removed (name: text)
Picture 150 (old this-work plot); Title 1 (x2, empty, at 14.5/13.7 and 0.49/12.07); AutoShape 10 (blank, 16.45/12.68); Picture 28 (Europa cutaway); TextBox 29 "Chemical Composition", 30 "Redox State", 31 "pH", 32 "Adiabatic Profile", 34 "Material Properties", 35 "Image: NASA/JPL-Caltech"; Straight Arrow Connectors 33, 39, 44, 90; Title 1 "Europa's Interior Properties", "Are Dictated by Its Ocean Chemistry"; TextBox 43 "Redox State logfH2", 45/46 "ice thickness (km)"; Pictures 47, 49 (density/conductivity maps); TextBox 48 "28% increase in ocean density...", 50 "Up to 10.8 S/m change...", 51 "ocean density", 52 "ocean conductivity", 53 "(Chang et. al, in prep)", 54 "Ice-Water Transition"; TextBox 55 (empty), 70 (empty); Group 171 (speciation panels, labels, captions, footnote, blank frame); Title 1 "Can Europa Clipper's Geophysical Investigation ..." ; Picture 81 (task flowchart); Rectangle 107 (blank); TextBox 109 "*" (stray); Straight Connectors 117, 118, 122 (green dividers); Right Arrow 142; TextBox 151 "New measurements in 5wt% NaCl..." (superseded); TextBox 165 "This can run on your phone"; TextBox 183 "Vance et al. (2018, 2021). https://github.com/vancesteven/PlanetProfile"; Picture 188 (PlanetProfile QR); Rectangle 133 (blank, inside Group 131). 44 removals, confirmed absent by text search (Chang, phone, Zolotov, Petricca, Neural, Compositional, PlanetProfile, Thermo/Orion, "Please add", Carbon, Europa's Interior).
Kept (present): logo, induction figure + its text, whole benchtop section (heading, instruments text, meter photo, circuit, "Measurements were conducted by", Table 1, melting-line diagram with liquid/ice/solid carbonate labels, MgSO4 and NaCl plots and labels, "Mahboub et al. 2026" + QR), compilation plot with its legend and "5wt% NaCl(aq)", "Psarakis et al. 2024", "Adams and Hall (1931) Experiments" legend overlay, Adams scan + "Adams and Hall 1931 (NaCl(aq))", Larionov scan + "Larionov 1984 for MgSO4(aq)" + axis labels, references box.

## Shapes re-worded / added (before -> after)
- Title (TextBox 83): "Neural Network-Based Geophysical Inference for Joint Inversion of Ocean World Mission Data" -> `TITLE` = "[PASTE AGU TITLE HERE]".
- Authors (TextBox 85): "Steven D. Vance1, Scott D. Chang2, Jesus Cortes3, Rose Garrity4, Zachary Diaz5, Leila Mahboub6, Flavio Petricca1, Baptiste Journaux2, Jessica M. Weber1" -> "Steven D. Vance1, Jesus Cortes2, Rose Garrity3, Zachary Diaz4, Jessica M. Weber1, Alicja Urbanska5, Leïla Mahboub6, Kristjan Stone1, Abby Kavner3".
- Affiliations (TextBox 87): rewritten to the six affiliations + email given in the brief.
- Induction text (Content Placeholder 2): wording unchanged; 24 pt, left aligned, no autofit (was centered with 44 pt space-before).
- Footnote (TextBox 114): "*Thermo/Orion Star A329 Image - Please add the following disclaimer: Reference herein ..." -> "*Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise, does not constitute or imply its endorsement by the United States Government or the Jet Propulsion Laboratory, California Institute of Technology." (14 pt, was 12 pt Roboto-Light).
- Benchtop heading (TextBox 94): wording unchanged, trailing empty paragraph dropped.
- Subtitle (TextBox 135): "Recent experiments and modeling show decreasing σ with P / (Pan et al. 2021, Psarakis et al. 2024)" kept, 18 -> 24 pt.
- Added: "Rev2 heading motivation", "Rev2 body motivation" (text as in brief), "Rev2 heading hero", Fig B, "Rev2 takeaway box" (26 pt, 3 paragraphs), "Rev2 heading new measurements" (40 pt bold + 26 pt line), Fig A, Fig C + caption, "Rev2 What we find box", "Rev2 Method box", "Rev2 photo asterisk", six "Rev2 separator" lines.
- Text changed from the brief: finding 1 is "In 0.70 mol/kg KCl the rise is 7-13%, on average about 2 percentage points above NaCl at the same temperature (0.9-2.6)." (facts f: 0.9-2.6 pp, mean 1.8); a trailing period was added to the last bullet.

## Numbers and their poster_facts.md sources
- 6% at 25-29 °C, 11% at 2-7 °C (0.9 mol/kg NaCl, to 200 MPa): section a / table: 6.23 (24.9), 5.84 (29.3), 6.64 (23.0); 10.62 (7.0), 11.22 (1.7). Verified, no change.
- "equals warming the ocean by 3-4 °C": section b, dT 3.0-4.0 K (MC12 3.1 at 24.8 °C, 3.8 at 1.7 °C, 4.0 at 7.0 °C). Verified.
- Pan et al. fit 212-1713 MPa; +1% to -10%: section d (-10.4% to +1.0% for 0.9 mol/kg). Verified.
- KCl 0.70 rise 7-13%: 7.09-12.71 (section a). "About 2 percentage points above NaCl": section f gives 0.9-2.6, mean 1.8; text amended to "on average about 2 (0.9-2.6)" so the stated number matches the facts.
- 0.5 mol/kg peaks near 200-250 MPa: highest-step 198-250, parabola 206-259; guide polynomial goes to 277 (as in rev1; author to decide). Decline to 700 MPa: table "% at highest P". As in Adams & Hall: section c says only the position of the maximum is consistent.
- 0.5 mol/kg 13-22% by 200 MPa: 12.9-21.9 excluding the 9.85 °C isotherm (31.5 %, flagged unsettled); "about twice": 2.1x (section h); A&H +6.6% at 25 °C (section c). Verified, with the 9.85 °C caveat.
- 0.34 percentage points (20-190 MPa; 0.9 NaCl and KCl, Nov 2026): maximum 0.34 (KCl 10.9 °C); NaCl Nov 0.20-0.32. The Oct 7-8 NaCl run is 0.80 and is not covered by "Nov 2026". Verified.
- Row 2 ranges: NaCl 0.5 to 700 MPa, 0.8-35 °C (T 0.85-34.85); 0.9 mol/kg 1.7-29 °C; KCl 6-30 °C (6.3-29.5), per rev1 check. Method: anchoring to McCleskey (2011) (below 5 °C McCleskey 2012 is used, see facts g).
- No number was corrected from the brief other than the amended "about 2 percentage points" wording.

## References (final list, alphabetical, 18 pt)
Kept: Adams & Hall 1931; Guo & Keppler 2019 (compilation legend); Kraichman 1964 "Defense Lab" (compilation legend, NaCl plot); Larionov & Kryukov 1984; McCleskey 2011; Pan et al. 2021; Psarakis et al. 2024; Sinmyo & Keppler 2016 (compilation legend); full entries copied from the draft.
Added from the abstract: Mahboub et al. 2026 (ACS Earth Space Chem, doi 10.1021/acsearthspacechem.5c00333); Roberts et al. 2023; Vance et al. 2023; Van Hoolst et al. 2024.
Added from the repo: McCleskey, Nordstrom, Ryan & Ball (2012), Geochim. Cosmochim. Acta 77:369-382 (from `docs/MCCLESKEY_METHOD.md` and `external/Jesus-Conductivity/references.bib`; no DOI in the repo).
**Incomplete (for the author):** "McDougall & Barker (2011, Gibbs Seawater) [full reference to add]" – the repo has only the cite key `mcdougall2011getting` (mahboub2026/main_revisions2.tex), no entry; the induction text says "(McDougall 2011)". Note the poster's induction text says "McDougall 2011" while the reference entry says "McDougall & Barker (2011)": please reconcile.
Dropped (no longer cited): Petricca et al. 2026, Zolotov & Kargel 2009, Vance et al. 2018, Vance et al. 2021. Sorting note: "Defense Lab: Kraichman" is filed under K; "Van Hoolst" is filed after "Vance".

## Verification results
- Re-opened rev2 with python-pptx: 1 slide, 44.0 x 44.0 in; 50 top-level shapes; no shape outside 0-44 in and none inside the 0.4 in outer margin; original md5 unchanged.
- Overlaps > 0.05 in^2 over all leaf shapes (groups resolved): 11, ALL intended overlays that exist in the draft: Picture 96 (Table 1 image, which contains its own copy of the melting-line diagram) under Group 99 Picture 100 (the melting-line diagram, 45.25 in^2) and the liquid/ice/solid carbonate labels on both (1.27, 1.27, 0.44, 0.44, 0.28, 0.28: labels on their own figure); compilation plot with its legend overlay "Adams and Hall (1931) Experiments" (1.89), "Psarakis et al. 2024" (1.47), "5wt% NaCl(aq)" (1.16) (labels on their own figure); Larionov scan with its "P (MPa)" axis label (0.45). The other 21 overlaps of the first build were resolved (Adams/Larionov scans vs compilation; stacked benchtop boxes; circuit vs diagram; photo vs text; TextBox 177/141 labels).
- Text fit (DejaVu Sans, 1.2 line height, >90 % = failure): all 18 new or re-set boxes pass (80-89 %; the table is printed by the script). The title is 51 % in its two-line box with the placeholder.
- Not checked in the fit: the kept benchtop text boxes TextBox 98/104/105/106 (their boxes were cut to text; with DejaVu in the preview "Benchtop conductivity sensor", "Thermo Scientific ..." lines are visibly tight, which is the substitute font's width; in the draft they were equally tight).
- Visual inspection of the four PNGs: reading order left-to-right (motivation, hero, benchtop; then new measurements; then Fig C / findings / method; earlier work in the right column; references); columns aligned to the gutters (0.5, 14.0, 26.48); no empty area larger than about 4 x 4 in except the right half of the title row (placeholder title is short) and the area x 26.5-30, y 7.5-10 beside the QR code (about 3.5 x 2.5 in).

## Open items for the author
1. Official AGU title: paste into `TITLE` (or `--title`); re-run. Check the real PowerPoint rendering of the title (size) and header height.
2. Fig C (7 x 6 in, from rev1) has an internal-note annotation "1-bar step unsettled?" at the 9.85 °C point, and "NaCl 0.5/0.9" labels; decide whether it stays on a public poster.
3. QR code target (31.1, 7.5) unverified (no decoder available); caption "Mahboub et al. 2026".
4. Table 1 lists NH4Cl and Na2CO3; the abstract names KCl and Na2SO4 (the poster has no Na2SO4 data shown).
5. Abstract says 274-318 K for NaCl; measured upper bound is 308 K (facts e). The poster says 0.8-35 °C. Abstract title/"confirms Adams and Hall" wording is not repeated; the poster states that only the position of the maximum agrees and the 0.5 mol/kg magnitude is about twice A&H's.
6. McDougall reference (above); McCleskey 2012 has no DOI in the repo.
7. 0.5 mol/kg maximum "200-250 MPa" ignores the guide-polynomial values (219-277 MPa).
8. Compilation plot: its in-image legend reads "Adams and Hall (1930)" while the overlay label says 1931; the figure title says "5wt% NaCl" (about 1 mol/kg) vs the 0.5 mol/kg A&H curve elsewhere. Pre-existing in the draft.
9. The meter photo (Thermo/Orion) is cropped 5.7 % at the bottom; the footnote sits left of it (see deviation 4).
10. Induction figure is 12.0 in wide; if Row 1 may extend to y about 23, it can be enlarged to 13.0 in (change `21.9` in the script).
