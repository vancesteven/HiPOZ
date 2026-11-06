# How HiPOZ computes McCleskey (2012) conductivity

Updated: 2026-11-05

This is the method behind every "MC12" curve and every Δ panel in the Cortes,
Mahboub and Garrity figures. Each step names the code that does it.

Reference: McCleskey, Nordstrom, Ryan & Ball (2012), *A new method of
calculating electrical conductivity with applications to natural waters*,
Geochim. Cosmochim. Acta 77, 369–382. The coefficient table is the one USGS
distributes (`conductivity-USGS.dat`, as used by Geochemist's Workbench).

## 1. Model

Conductivity is a sum over the **free, charged** species in solution:

    σ = Σ_i λ_i(T, I) · m_i

- m_i: free molality of species i (mol/kg H2O), from a WATEQ4F speciation
  (step 3). Neutral complexes such as MgSO4(aq) carry no current and are
  left out of the sum.
- λ_i: the molal conductivity of species i (S cm² mol⁻¹), given by a
  temperature polynomial and an ionic-strength correction:

      λ_i⁰(T) = λ2·T² + λ1·T + λ0
      A_i(T)  = A2·T² + A1·T + A0
      λ_i     = λ_i⁰(T) − A_i(T) · √I / (1 + B_i √I)

  with T in °C. λ2, λ1, λ0, A2, A1, A0 and B_i are tabulated per species.
- The sum is in mS/cm and is converted to S/m (× 0.1).

Code: `sigmaElectricMcCleskey2012.elecCondMcCleskey2012(T_C, ions)`.

### Ionic strength: one value for the whole solution

    I = ½ Σ_i m_i z_i²

summed over every charged species with parameters, at the same free
molalities as the conductivity sum. **Every λ_i is evaluated at this shared
I.** Before 2026-10-05 each ion used only its own partial ionic strength.
That under-corrected λ and read high by several percent at molal
concentrations: +4.6% against the 111.3 mS/cm KCl standard at 1 mol/kg.
Test: `tests/test_mccleskey_ionic_strength.py` (KCl standards within 1%).

### Coefficient table

`sigmaElectricMcCleskey2012.py` holds 28 species. Two kinds of correction
were made to the original transcription:

- Signs restored on lam2/A2 for H⁺, CO3²⁻, HCO3⁻, Fe³⁺ and KSO4⁻, and a
  missing zero restored on CO3²⁻ lam2 (−0.00326 → −0.000326). These match
  PlanetProfile commit b6a34e2.
- The large λ0 values of the charged ion pairs (NaSO4⁻ 219.2, KSO4⁻ 129.7,
  NaCO3⁻ 89.51 at 0 °C; about 357, 234 and 188 at 25 °C, against 77 for
  Cl⁻) were checked on 2026-10-05. They match the published USGS table
  coefficient for coefficient. They are **effective** parameters: MC12
  fitted them together with WATEQ4F speciation to whole-electrolyte data, so
  they mean something only when multiplied by WATEQ4F free molalities. Do
  not compare them with infinite-dilution single-ion mobilities.

How much the ion pairs matter (25 °C, speciated): NaSO4⁻ supplies 28–29% of
model σ for 0.25–0.5 mol/kg Na2SO4. In the Na2SO4:KCl mixtures NaSO4⁻ and
KSO4⁻ together supply 11–22%.

## 2. Composition → ion inputs

| Case | Ion inputs per mole | Code |
|---|---|---|
| Single salt | stoichiometric, e.g. Na2SO4 → 2 Na⁺ + SO4²⁻ | `study_plots.ION_SPECS` |
| Mixture `A:B_r1:r2` | per mole of the **first** component: A's ions + (r2/r1)·B's ions | `cortes_mccleskey.get_mixture_ion_spec` |

The concentration attached to a mixture is the first component's. For
example, `Na2SO4:KCl_2:1` at 1.5 means 1.5 Na2SO4 + 0.75 KCl. An earlier
version normalized the ratio to sum to 1 instead, which put the model at
1/2, 2/3 or 1/3 of the real concentrations (Na2SO4:KCl 1:1 RMS error
34% → 2.3% after the fix; `tests/test_cortes_concentrations.py`).

## 3. Speciation (WATEQ4F)

`speciation.py` runs Reaktoro's built-in PHREEQC engine with `wateq4f.dat`.
That is the database McCleskey et al. fitted against. It returns the free
molality of every charged species.

- Single salts with a recipe (`speciation.SALT_RECIPES`: NaCl, KCl, MgSO4,
  Na2SO4, NH4Cl, Na2CO3) use `speciated_ions_dict`. Mixtures are speciated
  from their ion inputs by `speciated_ions_from_spec`.
- Charged species that WATEQ4F forms but MC12 has no parameters for (e.g.
  MgOH⁺) are left out of the sum and reported once in a warning. Their
  molalities are tiny at near-neutral pH.
- For Na2CO3, the inputs fix the pH: dissolving solid Na2CO3 gives pH about
  11.4. See the carbonate note in `speciation.py`.
- Cross-check: Reaktoro and an independent USGS PHREEQC 3.8.6 +
  `wateq4f.dat` run agree at all six (T, m) points for Na2SO4, and for the
  NaCl:MgSO4 and Na2SO4:KCl mixtures (`tests/test_speciation.py`).

What speciation does (25 °C, 0.5 mol/kg):

| Solute | Full dissociation | Speciated | Why |
|---|---|---|---|
| NaCl | 4.540 S/m | 4.540 S/m | WATEQ4F has no Na–Cl pair |
| KCl (1 mol/kg) | 11.143 | 11.143 | no K–Cl pair |
| MgSO4 | 5.275 | 2.251 | 64% of Mg and SO4 is neutral MgSO4(aq); free Mg²⁺ = SO4²⁻ = 0.178 |

So the conductivity standards (KCl) and the NaCl calibration are unaffected.
Sulfate-bearing solutions are where speciation matters.

**When speciation runs.** The library default is
`compute_mccleskey_model(speciation=False)`, which keeps the old behaviour
when called directly. The Cortes pipeline uses
`cortes_mccleskey.resolve_speciation(compound, 'auto')`. That switches
speciation on whenever Reaktoro is installed and the solute has a recipe or
fully mappable ions. Without Reaktoro it falls back to full dissociation and
prints a note. To reproduce the paper figures, Reaktoro must be installed
(`regenerate_cortes_figures.sh` reports which mode it used).

## 4. Inputs from the measurements

- **Concentration in mol/kg H2O.** Benchtop solutions were made up in mol/L.
  `cortes2026/molarity_to_molality.py` converts them using solution
  densities from PHREEQC 3.8.6 + `pitzer.dat` at 20 °C (factors in
  `MOLALITY_TABLE`; the original mol/L is kept in `concentration_molar`).
  The conversion raises the concentrations by 1–5%.
- **Temperature.** The measured T of each point. Chiller-bath rows at 0 MPa
  use the bath setpoint (`analysis_config.bath_state`).
- **Pressure.** MC12 is a 1-bar model. It is compared only with
  ambient-pressure data. Gamry readings below 10 MPa are taken as 0.1 MPa
  (`cortes_data_processing.AMBIENT_P_CUTOFF_MPA`; the gauge value is kept in
  `P_MPa_raw`). The comparison uses P ≤ 5 MPa
  (`cortes_mccleskey.filter_low_pressure`), and the paper's EIS overlay
  uses P ≤ 1 MPa (`cortes2026_plots.EIS_PRESSURE_MAX`). High-pressure data
  are plotted but never compared with MC12.

## 5. Validity limit

MC12 is validated for effective ionic strength 0.0004–0.7 mol/kg, where I_eff
is the speciated value from step 1. Ion pairing lowers I_eff below its
nominal value, so sulfate-bearing solutions reach the limit at a higher
concentration than full dissociation would suggest. The limits
(first-component mol/kg, 5 °C, the strictest benchtop temperature) are in
`cortes_mccleskey.MCCLESKEY_LIMITS`. They were generated by
`cortes2026/mccleskey_validity_limits.py` with PHREEQC 3.8.6 +
`wateq4f.dat`:

| Solute | Limit (mol/kg) |
|---|---|
| NaCl, KCl | 0.700 |
| MgSO4 | 0.372 |
| Na2SO4 | 0.291 |
| Na2SO4:KCl 2:1 / 1:1 / 1:2 | 0.244 / 0.209 / 0.162 |
| NaCl:MgSO4 2:1 / 1:1 / 1:2 | 0.323 / 0.221 / 0.138 |

Figures mark the limit as a red dashed line, and the region beyond it is
shaded (`tests/test_mccleskey_validity.py`). Most Cortes mixture points lie
beyond the limit. Agreement or disagreement there tests extrapolation, not
the validated model.

## 6. Deviation

    Δ = (σ_data − σ_model) / σ_model   (in %)

Code: `cortes_mccleskey.compute_delta`. The axis label states this
definition (`tests/test_delta_label.py`).

## 7. Current agreement (Cortes benchtop, speciated, 2026-11-05)

RMS Δ by solute, from `cortes2026/prepost_comparison/prepost_summary.txt`.
The pre-fix column is before both the ionic-strength and the
mixture-concentration fixes.

| Solute | n | Pre-fix | Current |
|---|---|---|---|
| NaCl | 4 | 1.9% | 5.9% |
| KCl | 12 | 2.1% | 2.8% |
| Na2SO4 | 6 | 16.2% | 20.4% |
| Na2SO4:KCl 1:1 | 9 | 52.9% | 2.3% |
| Na2SO4:KCl 1:2 | 3 | 111.3% | 7.5% |
| Na2SO4:KCl 2:1 | 3 | 4.5% | 15.0% |
| NaCl:MgSO4 1:1 | 12 | 18.6% | 16.3% |
| NaCl:MgSO4 2:1 | 3 | 15.4% | 13.8% |
| NaCl:MgSO4 1:2 | 3 | 0.7% | 18.7% |

The low pre-fix numbers for some solutes came from errors cancelling: the
per-ion I bug read high while the mixture-concentration bug read low.

## Reproduce

```bash
conda install -c conda-forge reaktoro      # speciation
./regenerate_cortes_figures.sh             # all Cortes figures + prepost table
pytest tests/test_mccleskey_ionic_strength.py tests/test_speciation.py \
       tests/test_cortes_concentrations.py tests/test_mccleskey_validity.py
python validate_speciation.py --dataset both
```
