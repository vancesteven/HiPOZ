"""
Pan et al. (2021) electrical conductivity of aqueous NaCl at high pressure.

Pan, Y., Yong, W., Secco, R. A. (2021), Electrical conductivity of aqueous
NaCl at high pressure and low temperature: application to deep subsurface
oceans of icy moons, Geophys. Res. Lett. 48, e2021GL094020,
https://doi.org/10.1029/2021GL094020. Experiments: 212-1713 MPa, 233-295 K,
1/5/10 wt% NaCl.

The regression follows the authors' spreadsheet (Mendeley Data,
https://doi.org/10.17632/g43xkvm3gx.6, sheet 'regression'), as ported in
planetprofile-genai (PlanetProfile/Thermodynamics/NaCl/NaClProps.py, commit
bf545b5e); tests/test_pan2021_nacl.py checks it against the authors' 69
regression rows (tests/data/pan2021_validation.csv):

    log10(sigma) = A + B/T + C log10(chat) + D log10(rho)
                   + 2 log10(Lambda0) - log10(10 Lambda0 + 5584.2 chat_k)

with c the molality (mol/kg), rho the pure-water density (g/cm^3, SeaFreeze
'water1', as the paper uses), chat = c rho0 / (1 + M c) and
chat_k = c rho0 / sqrt(1 + M c). Outside the experimental envelope the
model is an extrapolation; below ~212 MPa it misses the conductivity maximum
the paper's MD simulations show.
"""
import numpy as np
from seafreeze.seafreeze import seafreeze

_A, _B, _C, _D = (-2.4857161222849897, 902.6050662042346,
                  1.7644668238775978, 1.5569422624900764)
_RHO0_GCM3 = 1.004205
_M_NACL_KGMOL = 0.05844
_KOHLRAUSCH = 5584.2

P_RANGE_MPA = (212.0, 1713.0)
T_RANGE_K = (233.0, 295.0)
MOLAL_MAX = 1.902


def sigma_Sm(molality, T_K, rho_gcm3):
    """Conductivity (S/m) from molality, T and pure-water density."""
    c = np.asarray(molality, dtype=float)
    T = np.asarray(T_K, dtype=float)
    rho = np.asarray(rho_gcm3, dtype=float)
    denom = 1.0 + _M_NACL_KGMOL * c
    chat = c * _RHO0_GCM3 / denom
    chat_k = c * _RHO0_GCM3 / np.sqrt(denom)
    lam0 = 3101.85 - 100.45 * rho - 1481175.59 / T + 185698255.2 / T ** 2
    return 10.0 ** (_A + _B / T + _C * np.log10(chat) + _D * np.log10(rho)
                    + 2.0 * np.log10(lam0)
                    - np.log10(10.0 * lam0 + _KOHLRAUSCH * chat_k))


def sigma_vs_P(molality, T_K, P_MPa):
    """Conductivity along an isotherm, water density from SeaFreeze."""
    P = np.atleast_1d(np.asarray(P_MPa, dtype=float))
    rho = seafreeze(np.array([P, np.array([float(T_K)])], dtype=object),
                    'water1').rho[:, 0] / 1e3
    return sigma_Sm(molality, T_K, rho)


def in_envelope(P_MPa, T_K):
    """True where (P, T) lies inside the experimental range."""
    P = np.asarray(P_MPa, dtype=float)
    return ((P >= P_RANGE_MPA[0]) & (P <= P_RANGE_MPA[1])
            & (T_RANGE_K[0] <= T_K <= T_RANGE_K[1]))
