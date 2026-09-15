#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WATEQ4F aqueous speciation for the McCleskey et al. (2012) conductivity model.

McCleskey et al. (2012) compute specific conductance as a sum over *free,
charged* aqueous species,

    sigma = SUM_i lambda_i(I, T) * m_i

where the m_i come from a geochemical speciation calculation (WATEQ4F), not from
analytical (total) salt molality. Neutral complexes such as MgSO4(aq) never
appear in the sum because they carry no charge -- that omission *is* the
"neutral species subtraction".

Two independent confirmations that the coefficient table in
sigmaElectricMcCleskey2012.py expects speciated input:

  * it carries lambda parameters for the *charged* complexes NaSO4-, KSO4-,
    NaCO3-, HSO4- and HCO3-, which only arise from a speciation calculation;
  * it has no entry at all for neutral MgSO4(aq).

Speciation is performed with Reaktoro's embedded PHREEQC implementation reading
the wateq4f.dat database that ships inside Reaktoro. This was cross-checked
against an independent phreeqpython + IPhreeqc 3.4.0 run and agreed to every
printed digit (MgSO4 free fraction 59.0% at 0.0249 molal and 25.7% at 1.6616
molal; ionic strength 0.0587 and 1.7056).

Reaktoro is an optional dependency (conda-forge only, not on PyPI). It is
imported lazily so that every code path which does not ask for speciation keeps
working without it.
"""

import logging
import numpy as np

log = logging.getLogger('HiPOZ')

# Reaktoro database name. Ships inside the reaktoro package -- no external path.
WATEQ4F_DB = 'wateq4f.dat'

# PHREEQC species name -> key in sigmaElectricMcCleskey2012.py's parameter table.
# Only charged species appear here; neutral species are excluded from the
# conductivity sum by construction (see module docstring).
PHREEQC_TO_MCCLESKEY = {
    'H+': 'H_p1',
    'Na+': 'Na_p1',
    'K+': 'K_p1',
    'Li+': 'Li_p1',
    'Cs+': 'Cs_p1',
    'NH4+': 'NH4_p1',
    'Ca+2': 'Ca_p2',
    'Mg+2': 'Mg_p2',
    'Ba+2': 'Ba_p2',
    'Sr+2': 'Sr_p2',
    'Al+3': 'Al_p3',
    'Cu+2': 'Cu_p2',
    'Fe+2': 'Fe_p2',
    'Fe+3': 'Fe_p3',
    'Mn+2': 'Mn_p2',
    'Zn+2': 'Zn_p2',
    'OH-': 'OH_m1',
    'Cl-': 'Cl_m1',
    'F-': 'F_m1',
    'Br-': 'Br_m1',
    'NO3-': 'NO3_m1',
    'SO4-2': 'SO4_m2',
    'CO3-2': 'CO3_m2',
    'HCO3-': 'HCO3_m1',
    'HSO4-': 'HSO4_m1',
    # WARNING -- provisional. The lambda_0(25 C) values for these three charged
    # complexes look implausible next to Cl- (77 S cm2/mol): NaSO4- 357 (4.6x),
    # KSO4- 234 (3.1x), NaCO3- 188 (2.4x). A singly-charged complex should sit
    # near or below Cl-. Because speciation is what first activates them, they
    # dominate the result wherever they form: NaCO3- alone supplies 51% of the
    # speciated conductivity of 0.3 molal Na2CO3 from just 0.19 molal, pushing
    # sigma *up* -- backwards for an association correction. An
    # equivalent-vs-molal convention error was ruled out (0.001 molal MgSO4
    # reproduces literature to -0.9% on molality, +92% on equivalents), so the
    # values are anomalous in the table itself. Verify against MC12's published
    # parameter table before trusting speciated sulfate/carbonate results.
    # MgSO4 is unaffected: it needs only the neutral MgSO4(aq) exclusion plus
    # Mg+2 and SO4-2, all of which check out.
    'NaSO4-': 'NaSO4_m1',
    'KSO4-': 'KSO4_m1',
    'NaCO3-': 'NaCO3_m1',
}

# Compounds whose speciated result is trustworthy today. The others speciate
# correctly but multiply free molalities by the suspect lambdas flagged above.
VERIFIED_COMPOUNDS = ('MgSO4', 'NaCl', 'KCl', 'NH4Cl')

# Salt -> (elements to speciate, {input ion: moles per mole of salt}).
#
# Carbonate note: for Na2CO3 the *input specification* fixes the pH, which then
# dominates carbonate speciation. Supplying Na+ and CO3-2 gives the alkaline
# solution you get by dissolving solid Na2CO3 (pH ~11.4 at 0.3 molal, carbonate
# largely as CO3-2 and NaCO3-); supplying Na+ and HCO3- instead gives a
# near-neutral solution (pH ~7.7, carbonate almost entirely HCO3-). Both are
# self-consistent -- they are different solutions. We specify the carbonate
# input explicitly here so the choice is visible rather than implicit. These
# systems are treated as closed to atmospheric CO2.
SALT_RECIPES = {
    'NaCl':   ('Na Cl O H',    {'Na+': 1.0, 'Cl-': 1.0}),
    'KCl':    ('K Cl O H',     {'K+': 1.0, 'Cl-': 1.0}),
    'MgSO4':  ('Mg S O H',     {'Mg+2': 1.0, 'SO4-2': 1.0}),
    'Na2SO4': ('Na S O H',     {'Na+': 2.0, 'SO4-2': 1.0}),
    'NH4Cl':  ('N Cl O H',     {'NH4+': 1.0, 'Cl-': 1.0}),
    'Na2CO3': ('Na C O H',     {'Na+': 2.0, 'CO3-2': 1.0}),
}

# Speciating elemental S or N lets Reaktoro form reduced species (S4-2, S5-2,
# HS-, NH4+ from NO3- ...) that cannot occur in an oxidised benchtop solution
# prepared from a sulfate or ammonium salt. They come back at ~1e-20 molal and
# are harmless numerically, but they trip the "no McCleskey parameters" warning
# on every call, so drop anything below this molality before reporting.
NEGLIGIBLE_MOLAL = 1e-12

# Cache keyed on (compound, rounded molality, rounded T_K): a Reaktoro
# equilibrium solve per point is far too slow to repeat inside plot loops.
_CACHE = {}

# Charged species seen without McCleskey parameters, warned about once each.
_WARNED_MISSING = set()

_RKT = None
_SYSTEMS = {}


def _reaktoro():
    """Import Reaktoro on first use and silence its non-fatal warnings."""
    global _RKT
    if _RKT is None:
        try:
            import reaktoro as rkt
        except ImportError as e:
            raise ImportError(
                'Speciation requires Reaktoro, which is distributed via '
                'conda-forge only (not PyPI): conda install -c conda-forge '
                'reaktoro. Use speciation=False to run the McCleskey model on '
                'total molality instead.') from e
        # 906: chemical convergence warnings, handled via solve().succeeded().
        # 548: "ionic strength > 6 molal" -- raised for intermediate solver
        #      iterates, not the converged state, so it is noise here.
        rkt.Warnings.disable(906)
        rkt.Warnings.disable(548)
        _RKT = rkt
    return _RKT


def _system(compound):
    """Build (and cache) the Reaktoro chemical system for one salt."""
    if compound not in _SYSTEMS:
        rkt = _reaktoro()
        if compound not in SALT_RECIPES:
            raise ValueError(
                f'No speciation recipe for {compound!r}. '
                f'Known: {sorted(SALT_RECIPES)}')
        elements, _ = SALT_RECIPES[compound]
        db = rkt.PhreeqcDatabase(WATEQ4F_DB)
        phase = rkt.AqueousPhase(rkt.speciate(elements))
        phase.set(rkt.ActivityModelPhreeqc(db))
        _SYSTEMS[compound] = rkt.ChemicalSystem(db, phase)
    return _SYSTEMS[compound]


def free_ion_molalities(compound, m_total, T_K, P_MPa=0.1):
    """
    Speciate a binary salt solution and return free charged-ion molalities.

    Parameters
    ----------
    compound : str
        Salt name, one of SALT_RECIPES (e.g. 'MgSO4', 'Na2CO3').
    m_total : float
        Total (analytical) salt molality, mol/kg water.
    T_K : float
        Temperature in K.
    P_MPa : float, optional
        Pressure in MPa. Default 0.1 MPa (1 bar). Speciation is only weakly
        pressure dependent over the benchtop range.

    Returns
    -------
    dict
        {mccleskey_ion_key: free_molality}, containing charged species only.
        Neutral complexes (e.g. MgSO4(aq)) are absent by design. Returns an
        empty dict for m_total <= 0.
    """
    m_total = float(m_total)
    T_K = float(T_K)
    if m_total <= 0:
        return {}

    key = (compound, round(m_total, 10), round(T_K, 6), round(float(P_MPa), 6))
    if key in _CACHE:
        return _CACHE[key]

    rkt = _reaktoro()
    system = _system(compound)
    _, inputs = SALT_RECIPES[compound]

    state = rkt.ChemicalState(system)
    state.temperature(T_K, 'kelvin')
    state.pressure(float(P_MPa), 'MPa')
    state.set('H2O', 1.0, 'kg')
    for ion, per_mole in inputs.items():
        state.set(ion, m_total * per_mole, 'mol')

    result = rkt.EquilibriumSolver(system).solve(state)
    if not result.succeeded():
        log.warning(
            f'Speciation did not converge for {compound} at {m_total:.4g} molal, '
            f'{T_K:.2f} K; falling back to total molality for this point.')
        _CACHE[key] = None
        return None

    props = rkt.AqueousProps(state)
    ions = {}
    for species in system.species():
        name = species.name()
        # Charge test: this is the neutral-species exclusion.
        if '+' not in name and '-' not in name:
            continue
        molality = float(props.speciesMolality(name))
        if molality <= NEGLIGIBLE_MOLAL:
            continue
        mc_key = PHREEQC_TO_MCCLESKEY.get(name)
        if mc_key is None:
            if name not in _WARNED_MISSING:
                _WARNED_MISSING.add(name)
                log.warning(
                    f'Charged species {name} has no McCleskey (2012) '
                    f'parameters; excluded from the conductivity sum.')
            continue
        ions[mc_key] = ions.get(mc_key, 0.0) + molality

    _CACHE[key] = ions
    return ions


def speciated_ions_dict(compound, concs_molal, T_K, P_MPa=0.1):
    """
    Build an `ions` dict for elecCondMcCleskey2012 from speciated molalities.

    Mirrors the shape that study_plots.compute_mccleskey_model builds from total
    molality: {ion_key: {'mols': ndarray}} with one entry per concentration.
    Concentrations whose speciation failed to converge fall back to the
    unspeciated stoichiometric molality so a single bad point cannot void a
    whole curve.

    Parameters
    ----------
    compound : str
        Salt name, one of SALT_RECIPES.
    concs_molal : array-like
        Total salt molalities, mol/kg water.
    T_K : float
        Temperature in K.
    P_MPa : float, optional
        Pressure in MPa.

    Returns
    -------
    dict
        {mccleskey_ion_key: {'mols': ndarray}} suitable for
        elecCondMcCleskey2012.
    """
    concs = np.atleast_1d(np.asarray(concs_molal, dtype=float))
    per_conc = []
    for m in concs:
        ions = free_ion_molalities(compound, m, T_K, P_MPa=P_MPa)
        if ions is None:
            # Non-convergent point: use stoichiometric (fully dissociated) values.
            _, inputs = SALT_RECIPES[compound]
            ions = {PHREEQC_TO_MCCLESKEY[ion]: m * mult
                    for ion, mult in inputs.items()
                    if ion in PHREEQC_TO_MCCLESKEY}
        per_conc.append(ions)

    keys = sorted({k for ions in per_conc for k in ions})
    return {k: {'mols': np.array([ions.get(k, 0.0) for ions in per_conc])}
            for k in keys}


def free_fraction(compound, m_total, T_K, ion=None, P_MPa=0.1):
    """
    Fraction of a salt's cation (or a named ion) remaining as a free ion.

    Diagnostic helper for validation output. Returns NaN if speciation failed.
    """
    ions = free_ion_molalities(compound, m_total, T_K, P_MPa=P_MPa)
    if not ions:
        return float('nan')
    _, inputs = SALT_RECIPES[compound]
    if ion is None:
        # Default to the first input ion (the cation for every recipe here).
        ion = next(iter(inputs))
    mc_key = PHREEQC_TO_MCCLESKEY.get(ion)
    if mc_key is None or mc_key not in ions:
        return float('nan')
    return ions[mc_key] / (m_total * inputs[ion])


def available():
    """True if Reaktoro can be imported (i.e. speciation is usable)."""
    try:
        _reaktoro()
        return True
    except ImportError:
        return False
