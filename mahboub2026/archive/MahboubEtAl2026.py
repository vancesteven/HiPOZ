#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import importlib.util

# ==========================================
# Global style / constants
# ==========================================
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["font.size"] = 14
plt.rcParams["pdf.fonttype"] = 42   # embed as TrueType (editable in Illustrator)
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["svg.fonttype"] = "none"  # keep SVG text as text
# chosen_map = "Set3" # pastels, yellow is index 1 and needs to be changed
chosen_map = "tab10" #


AxLabelSize = 18
TLabelSize = 10
LineWidth = 2
SAVE_FIGS = True

# McCleskey applicability “limits” (mol/kg_H2O)
MC12NaCl_limit   = 0.9999
MC12MgSO4_limit  = 0.01245
MC12NH4Cl_limit  = 1.034
MC12NaCO3_limit  = 0.3041

# ==========================================
# Helpers
# ==========================================
def pdiff(x, y):
    """Percent difference 100*(x - y)/y with NaN-safe behavior."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y != 0)
    out[mask] = 100.0 * (x[mask] - y[mask]) / y[mask]
    return out

def safe_errorbar(ax, x, y, yerr=None, **kwargs):
    """Trim x/y to a common length and scalarize mismatched yerr to avoid size errors."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if yerr is None:
        return ax.errorbar(x, y, **kwargs)
    ye = np.asarray(yerr).ravel()
    if ye.size == 1:
        yerr_use = float(ye.item())
    elif ye.size == n:
        yerr_use = ye
    else:
        yerr_use = 0.05 * (np.nanmean(np.abs(y)) if np.isfinite(y).any() else 1.0)
    return ax.errorbar(x, y, yerr=yerr_use, **kwargs)
#
# def annotate_right(ax, ys_raw, labels, *,
#                    fontsize=10, x_pad_frac=0.02,   # extra space on right
#                    min_px_gap=24, n_iter=60, k_push=0.25,
#                    connector=True):
#     """
#     Place right-edge labels with non-overlap and auto xlim padding.
#
#     x_pad_frac: fraction of current x-span added to the right
#     min_px_gap: minimum vertical gap between labels (in screen pixels)
#     """
#     # 1) Extend xlim to make room for labels
#     x0, x1 = ax.get_xlim()
#     span = (x1 - x0) if x1 > x0 else 1.0
#     x1_new = x1 + x_pad_frac * span
#     ax.set_xlim(x0, x1_new)
#     # label x-position slightly inside the new right edge
#     x_lab = x1 + 0.93* (x1_new - x1)
#
#     # 2) Clean/collect y endpoints
#     ys = np.array([np.nan if not np.isfinite(y) else float(y) for y in ys_raw])
#     idx = np.where(np.isfinite(ys))[0]
#     if idx.size == 0:
#         return
#
#     # 3) Convert min_px_gap to data units
#     inv = ax.transData.inverted()
#     y0_pix = ax.transData.transform((0, ax.get_ylim()[0]))[1]
#     y1_pix = y0_pix + min_px_gap
#     min_dy = abs(inv.transform((0, y1_pix))[1] - ax.get_ylim()[0])
#
#     # 4) Sort → repulse → unsort
#     order = np.argsort(ys[idx])
#     ywork = ys[idx][order].copy()
#     for _ in range(n_iter):
#         moved = False
#         for j in range(1, ywork.size):
#             gap = ywork[j] - ywork[j-1]
#             if gap < min_dy:
#                 push = (min_dy - gap) * k_push
#                 ywork[j]   += push
#                 ywork[j-1] -= push
#                 moved = True
#         if not moved:
#             break
#
#     ys_out = ys.copy()
#     ys_out[idx[order]] = ywork
#
#     # 5) Draw connectors and text (keep text editable; no clipping)
#     for y0, y1, s in zip(ys, ys_out, labels):
#         if not np.isfinite(y1):
#             continue
#         if connector and np.isfinite(y0):
#             ax.plot([x1, x_lab * 1], [y0, y1], lw=0.8, ls=":", color="0.4", zorder=2.5)
#         ax.text(x_lab, y1, s, ha="left", va="center",
#                 fontsize=fontsize, color="k", clip_on=False, zorder=3)

def annotate_right(ax, ys_raw, labels, *,
                   fontsize=10,
                   x_pad_frac=0.08,
                   x_text_rel=1.005,
                   x_conn_rel=1.03,
                   min_px_gap=24, n_iter=60, k_push=0.25,
                   connector=True):

    # find rightmost plotted x
    x_end = None
    for line in ax.lines:
        xd = np.asarray(line.get_xdata(), float)
        xd = xd[np.isfinite(xd)]
        if xd.size:
            x_end = xd.max() if x_end is None else max(x_end, xd.max())

    if x_end is None:
        x_end = ax.get_xlim()[1]

    # extend xlim
    x0, x1 = ax.get_xlim()
    span = x1 - x0 if x1 > x0 else 1.0
    x1_new = x1 + x_pad_frac * span
    ax.set_xlim(x0, x1_new)

    span_new = x1_new - x0
    x_text = x0 + x_text_rel * span_new
    x_conn = x0 + x_conn_rel * span_new

    ys = np.array([np.nan if not np.isfinite(y) else float(y) for y in ys_raw])
    idx = np.where(np.isfinite(ys))[0]
    if idx.size == 0:
        return

    # pixel → data spacing
    inv = ax.transData.inverted()
    y0_pix = ax.transData.transform((0, ax.get_ylim()[0]))[1]
    min_dy = abs(inv.transform((0, y0_pix + min_px_gap))[1] - ax.get_ylim()[0])

    order = np.argsort(ys[idx])
    ywork = ys[idx][order].copy()

    for _ in range(n_iter):
        moved = False
        for j in range(1, ywork.size):
            gap = ywork[j] - ywork[j-1]
            if gap < min_dy:
                push = (min_dy - gap) * k_push
                ywork[j]   += push
                ywork[j-1] -= push
                moved = True
        if not moved:
            break

    ys_out = ys.copy()
    ys_out[idx[order]] = ywork

    for y0, y1, s in zip(ys, ys_out, labels):
        if not np.isfinite(y1):
            continue
        if connector and np.isfinite(y0):
            ax.plot([x_end, x_conn], [y0, y1],
                    lw=1.8, ls=":", color="0.4", zorder=2.5)
        ax.text(x_text, y1, s, ha="left", va="center",
                fontsize=fontsize, clip_on=False, zorder=3)



def tight_xlim(ax, conc, pad=0.03, pad_left=None, pad_right=None):
    """
    Set xlim snug to data but allow asymmetric padding.
    pad: symmetric fraction if left/right not given
    pad_left, pad_right: fractions of span added to each side
    """
    c = np.asarray(conc, dtype=float).ravel()
    c = c[np.isfinite(c)]
    if c.size == 0:
        return
    cmin, cmax = np.min(c), np.max(c)
    if cmin == cmax:
        span = max(1e-3, abs(cmax))
        pl = pad_left if pad_left is not None else pad
        pr = pad_right if pad_right is not None else pad
        ax.set_xlim(cmin - pl*span, cmax + pr*span)
    else:
        span = cmax - cmin
        pl = pad_left if pad_left is not None else pad
        pr = pad_right if pad_right is not None else pad
        ax.set_xlim(cmin - pl*span, cmax + pr*span)

def molal_to_g_per_kg_solution(m, molar_mass):
    """
    Convert molality (mol/kg solvent) to g/kg solution.

    Parameters
    ----------
    m : float or array-like
        Molality [mol/kg solvent]
    molar_mass : float
        Molar mass [g/mol]

    Returns
    -------
    g_per_kg_solution : float or ndarray
        Grams of solute per kg of solution
    """
    m = np.asarray(m, dtype=float)
    return (m * molar_mass) / (1.0 + (m * molar_mass) / 1000.0)

# --------------------------
# Load McCleskey model file sitting next to this script
# --------------------------
MC_PATH = os.path.join(os.path.dirname(__file__), "sigmaElectricMcCleskey2012.py")
spec = importlib.util.spec_from_file_location("sigmaElectricMcCleskey2012", MC_PATH)
mc_mod = importlib.util.module_from_spec(spec)
sys.modules["sigmaElectricMcCleskey2012"] = mc_mod
spec.loader.exec_module(mc_mod)
elecCondMcCleskey2012 = mc_mod.elecCondMcCleskey2012

def mc_sigma(concs, T_list_C, ionspec, P_MPa=0.1):
    """
    concs   : (N,) mol/kg_H2O
    T_list_C: (M,) temperatures in Celsius
    ionspec : dict {'Na_p1': stoich, ...} values multiply concs
    return  : (N, M) sigma (S/m)
    """
    concs = np.asarray(concs, float)
    T_list_C = np.asarray(T_list_C, float)
    out = np.zeros((len(concs), len(T_list_C)), float)
    for j, tC in enumerate(T_list_C):
        ions = {ion: {"mols": concs * mult} for ion, mult in ionspec.items()}
        Tvec = np.full_like(concs, float(tC))
        out[:, j] = elecCondMcCleskey2012(P_MPa, Tvec, ions)
    return out


# ==========================================
# Data: NaCl
# ==========================================
conc_nacl = np.array([10, 30, 50, 75, 100, 150]) / 58.44
T_C_nacl = [-10, -6, -3, -1, 5, 20, 25]
tempsK_nacl = [t + 273.15 for t in T_C_nacl]
cmap = cm.get_cmap(chosen_map)
global_cmap = cmap(np.arange(7))

elec_cond_nacl_25 = 0.1*np.array([
    [17.88, 48.61, 74.90, 105.67, 132.33, 176.55],
    [17.77, 48.75, 75.01, 105.52, 132.77, 176.33],
    [17.53, 48.85, 75.91, 104.95, 132.55, 175.23]
])
elec_cond_nacl_20 = 0.1*np.array([
    [16.25, 44.19, 68.09, 96.06, 120.3, 160.5],
    [16.15, 44.32, 68.19, 95.93, 120.7, 160.3],
    [15.94, 44.41, 69.01, 95.41, 120.5, 159.3]
])
elec_cond_nacl_5 = 0.1*np.array([
    [11.88, 31.24, 47.89, 68.13, 85.99, 112.8],
    [12.01, 31.19, 48.23, 69.25, 86.10, 114.9],
    [11.57, 31.60, 49.40, 67.83, 87.87, 112.5]
])
elec_cond_nacl_minus1 = 0.1*np.array([
    [9.843, 26.97, 40.33, 57.44, 71.01, 94.74],
    [9.862, 26.49, 42.00, 56.78, 71.08, 95.81],
    [9.835, 27.51, 41.08, 56.64, 73.71, 95.80]
])
elec_cond_nacl_minus3 = 0.1*np.array([
    [10.33, 25.48, 38.18, 54.68, 70.74, 93.73],
    [9.593, 25.32, 40.45, 54.73, 69.13, 92.16],
    [9.547, 24.35, 39.69, 54.97, 70.31, 91.05]
])
elec_cond_nacl_minus6 = 0.1*np.array([
    [8.755, 23.62, 35.53, 50.67, 62.60, 84.79],
    [8.6485,23.89, 35.04, 50.89, 64.00, 85.79],
    [8.542, 23.09, 35.89, 51.86, 64.85, 85.29]
])
elec_cond_nacl_minus10 = 0.1*np.array([
    [7.852, 22.41, 32.26, 43.24, 57.06, 75.76],
    [7.852, 21.95, 31.87, 45.55, 56.45, 79.25],
    [7.852, 20.87, 31.33, 44.73, 58.90, 74.91]
])

def mean_std(a):
    return np.nanmean(a, axis=0), np.nanstd(a, axis=0)

elec_cond_mean_nacl_25, err_nacl_25 = mean_std(elec_cond_nacl_25)
elec_cond_mean_nacl_20, err_nacl_20 = mean_std(elec_cond_nacl_20)
elec_cond_mean_nacl_5,  err_nacl_5  = mean_std(elec_cond_nacl_5)
elec_cond_mean_nacl_minus1, err_nacl_minus1 = mean_std(elec_cond_nacl_minus1)
elec_cond_mean_nacl_minus3, err_nacl_minus3 = mean_std(elec_cond_nacl_minus3)
elec_cond_mean_nacl_minus6, err_nacl_minus6 = mean_std(elec_cond_nacl_minus6)
elec_cond_mean_nacl_minus10, err_nacl_minus10 = mean_std(elec_cond_nacl_minus10)

# total errors (std ⊕ 0.5% ⊕ 2.89% ⊕ 0.5%)
def total_err(std, mean):
    return np.sqrt(std**2 + (0.005*mean)**2 + (0.0289*mean)**2 + (0.005*mean)**2)

err_nacl_25      = total_err(err_nacl_25,      elec_cond_mean_nacl_25)
err_nacl_20      = total_err(err_nacl_20,      elec_cond_mean_nacl_20)
err_nacl_5       = total_err(err_nacl_5,       elec_cond_mean_nacl_5)
err_nacl_minus1  = total_err(err_nacl_minus1,  elec_cond_mean_nacl_minus1)
err_nacl_minus3  = total_err(err_nacl_minus3,  elec_cond_mean_nacl_minus3)
err_nacl_minus6  = total_err(err_nacl_minus6,  elec_cond_mean_nacl_minus6)
err_nacl_minus10 = total_err(err_nacl_minus10, elec_cond_mean_nacl_minus10)

# Gamry NaCl @ ~20 C
GamryNaCl_20C_20250813 = np.array([
    [0.5, 4.698607239],
    [0.5, 4.693751602],
    [0.5, 4.688398383],
    [1.0, 8.277955433],
    [1.0, 8.280132336],
    [1.0, 8.274678270]
])
conc_gamNaCl = GamryNaCl_20C_20250813[:, 0] # molal
# conc_gamNaCl_ppt = molal_to_g_per_kg_solution(conc_gamNaCl_molal,58.44)
elec_gamNaCl = GamryNaCl_20C_20250813[:, 1]
errGamNaCl = np.sqrt((4.0)**2 + (7.0*0.5)**2)  # %
errGamNaCl_scalar = 0.01 * errGamNaCl  # convert to fractional for plotting with S/m

# McCleskey model for NaCl
sigma_nacl_model = mc_sigma(
    conc_nacl, T_C_nacl,
    {"Na_p1": 1.0, "Cl_m1": 1.0}
)

vars_exp_nacl = [
    elec_cond_mean_nacl_minus10, elec_cond_mean_nacl_minus6, elec_cond_mean_nacl_minus3,
    elec_cond_mean_nacl_minus1,  elec_cond_mean_nacl_5,      elec_cond_mean_nacl_20,
    elec_cond_mean_nacl_25
]
vars_mod_nacl = [sigma_nacl_model[:,i] for i in range(7)]
errs_nacl = [
    err_nacl_minus10, err_nacl_minus6, err_nacl_minus3,
    err_nacl_minus1,  err_nacl_5,      err_nacl_20,
    err_nacl_25
]
temp_labels_nacl = ['263.15 K','267.15 K','270.15 K','272.15 K','278.15 K','293.15 K','298.15 K']

MC2011_NaCl = np.array([
    [0.000104, 0.008271, 0.009431, 0.01295, 0.01610, 0.01893, 0.02693, 0.03358],
    [0.000990, 0.07531,  0.08619,  0.1224,  0.1478,  0.1747,  0.2497,  0.3126 ],
    [0.01000,  0.7349,   0.8395,   1.185,   1.425,   1.694,   2.418,   3.029  ],
    [0.10000,  6.588,    7.505,    10.60,   12.77,   15.05,   21.48,   26.91  ],
    [0.5000,   29.48,    33.48,    46.52,   56.25,   66.24,   93.50,   116.8  ],
    [0.9999,   54.44,    61.23,    84.30,   101.9,   119.4,   168.0,   209.4  ],
], dtype=float)

MC2011_temps = [5.0, 10.0, 25.0, 35.0, 45.0, 70.0, 90.0]
MC2011_conc_molal = MC2011_NaCl[:, 0] # in molal
# MC2011_conc = molal_to_g_per_kg_solution(MC2011_conc_molal,58.44)
MC2011_vals = MC2011_NaCl[:, 1:]

# ==========================================
# Data: MgSO4
# ==========================================
conc_mgso4 = np.array([3, 40, 80, 120, 170, 200]) / 120.366
T_C_mgso4 = [-10, -6, -3, -1, 5, 20, 25]
tempsK_mgso4 = [t + 273.15 for t in T_C_mgso4]

elec_cond_mgso4_25 = 0.1*np.array([
    [3.318, 25.01, 40.03, 49.91, 56.52, 57.99],
    [3.443, 25.26, 40.04, 50.15, 56.33, 59.28],
    [3.508, 25.44, 40.11, 46.48, 56.56, 57.79]
])
elec_cond_mgso4_20 = 0.1*np.array([
    [3.016, 22.74, 36.39, 45.37, 51.38, 52.72],
    [3.130, 22.96, 36.40, 45.59, 51.21, 53.89],
    [3.189, 23.13, 36.46, 42.25, 51.42, 52.54]
])
elec_cond_mgso4_5 = 0.1*np.array([
    [2.245, 16.16, 25.05, 31.01, 35.43, 35.71],
    [2.126, 15.89, 25.31, 31.40, 34.77, 35.29],
    [2.255, 15.93, 25.20, 30.92, 34.80, 35.29]
])
elec_cond_mgso4_minus1 = 0.1*np.array([
    [1.997, 14.26, 21.00, 26.26, 28.97, 29.86],
    [1.836, 13.55, 21.01, 26.53, 29.32, 29.89],
    [1.958, 13.55, 21.42, 26.17, 29.20, 28.70]
])
elec_cond_mgso4_minus3 = 0.1*np.array([
    [1.190, 13.10, 19.80, 24.52, 27.80, 27.75],
    [1.753, 12.57, 18.60, 24.73, 26.63, 27.70],
    [2.106, 12.72, 19.62, 24.89, 28.69, 27.85]
])
elec_cond_mgso4_minus6 = 0.1*np.array([
    [1.843, 12.05, 18.12, 22.75, 24.42, 24.01],
    [1.737, 11.65, 18.47, 22.20, 23.87, 25.08],
    [2.086, 11.47, 18.04, 21.84, 24.88, 25.05]
])
elec_cond_mgso4_minus10 = 0.1*np.array([
    [0.000, 11.09, 15.50, 20.37, 21.88, 22.31],
    [0.000, 11.09, 15.50, 20.37, 21.87, 21.98],
    [0.000, 11.09, 15.50, 20.37, 21.61, 22.77]
])

elec_cond_mean_mgso4_25, err_mgso4_25 = mean_std(elec_cond_mgso4_25)
elec_cond_mean_mgso4_20, err_mgso4_20 = mean_std(elec_cond_mgso4_20)
elec_cond_mean_mgso4_5, err_mgso4_5 = mean_std(elec_cond_mgso4_5)
elec_cond_mean_mgso4_minus1, err_mgso4_minus1 = mean_std(elec_cond_mgso4_minus1)
elec_cond_mean_mgso4_minus3, err_mgso4_minus3 = mean_std(elec_cond_mgso4_minus3)
elec_cond_mean_mgso4_minus6, err_mgso4_minus6 = mean_std(elec_cond_mgso4_minus6)
elec_cond_mean_mgso4_minus10, err_mgso4_minus10 = mean_std(elec_cond_mgso4_minus10)

err_mgso4_25      = total_err(err_mgso4_25,      elec_cond_mean_mgso4_25)
err_mgso4_20      = total_err(err_mgso4_20,      elec_cond_mean_mgso4_20)
err_mgso4_5       = total_err(err_mgso4_5,       elec_cond_mean_mgso4_5)
err_mgso4_minus1  = total_err(err_mgso4_minus1,  elec_cond_mean_mgso4_minus1)
err_mgso4_minus3  = total_err(err_mgso4_minus3,  elec_cond_mean_mgso4_minus3)
err_mgso4_minus6  = total_err(err_mgso4_minus6,  elec_cond_mean_mgso4_minus6)
err_mgso4_minus10 = total_err(err_mgso4_minus10, elec_cond_mean_mgso4_minus10)

# Gamry MgSO4 @ ~19.5 C
GamryMgSO4_19p5C_molal_Spm = np.array([
    [0.5, 3.326447869],
    [0.5, 3.329707743],
    [0.5, 3.327332299],
    [0.75, 4.168767239],
    [0.75, 4.167934553],
    [0.75, 4.170431360],
    [1.0, 4.880327469],
    [1.0, 4.879407665],
    [1.0, 4.873111005],
    [1.5, 5.593752540],
    [1.5, 5.596119290],
    [1.5, 5.591247573]
])
conc_gamMgSO4  = GamryMgSO4_19p5C_molal_Spm[:, 0]
elec_gamMgSO4  = GamryMgSO4_19p5C_molal_Spm[:, 1]
errGamMgSO4 = np.sqrt((4.0)**2 + (5.0*1.0)**2)  # %
errGamMgSO4_scalar = 0.01 * errGamMgSO4

# McCleskey model for MgSO4
sigma_mgso4_model = mc_sigma(
    conc_mgso4, T_C_mgso4,
    ionspec={"Mg_p2": 1.0, "SO4_m2": 1.0}
)

vars_exp_mgso4 = [
    elec_cond_mean_mgso4_minus10, elec_cond_mean_mgso4_minus6, elec_cond_mean_mgso4_minus3,
    elec_cond_mean_mgso4_minus1,  elec_cond_mean_mgso4_5,      elec_cond_mean_mgso4_20,
    elec_cond_mean_mgso4_25
]

# enforce NaN at coldest T / first conc if exp value is zero (exclude from Δ)
if len(vars_exp_mgso4[0]) > 0 and np.isfinite(vars_exp_mgso4[0][0]) and abs(vars_exp_mgso4[0][0]) < 1e-12:
    v = np.asarray(vars_exp_mgso4[0], dtype=float).copy()
    v[0] = np.nan
    vars_exp_mgso4[0] = v

vars_mod_mgso4 = [sigma_mgso4_model[:,i] for i in range(7)]
errs_mgso4 = [
    err_mgso4_minus10, err_mgso4_minus6, err_mgso4_minus3,
    err_mgso4_minus1,  err_mgso4_5,      err_mgso4_20,
    err_mgso4_25
]
temp_labels_mgso4 = ['263.15 K','267.15 K','270.15 K','272.15 K','278.15 K','293.15 K','298.15 K']

# ==========================================
# Data: NH4Cl
# ==========================================
conc_nh4cl = np.array([10, 50, 75, 100]) / 53.491
T_C_nh4cl = [-10, -6, -3, -1, 5, 20, 25]
tempsK_nh4cl = [t + 273.15 for t in T_C_nh4cl]

elec_cond_nh4cl_25 = np.array([
    [2.3331, 10.3466, 14.938, 19.261],
    [2.3177, 10.3642, 14.894, 19.030],
    [2.3320, 10.4170, 14.718, 19.085]
])
elec_cond_nh4cl_20 = 0.1*np.array([
    [21.21, 94.06, 135.8, 175.1],
    [21.07, 94.22, 135.4, 173.0],
    [21.20, 94.70, 133.8, 173.5]
])
elec_cond_nh4cl_5 = 0.1*np.array([
    [15.35, 68.43, 100.0, 131.3],
    [15.37, 69.79, 99.93, 130.9],
    [15.76, 69.04, 99.21, 129.4]
])
elec_cond_nh4cl_minus1 = 0.1*np.array([
    [13.55, 58.97, 86.56, 115.4],
    [13.12, 60.68, 85.68, 113.3],
    [13.36, 59.83, 85.84, 109.2]
])
elec_cond_nh4cl_minus3 = 0.1*np.array([
    [12.53, 54.91, 81.88, 107.9],
    [12.58, 56.64, 82.93, 104.6],
    [12.70, 56.02, 82.03, 106.7]
])
elec_cond_nh4cl_minus6 = 0.1*np.array([
    [12.24, 52.07, 77.13, 100.4],
    [11.40, 54.28, 76.53, 98.49],
    [11.99, 51.78, 75.50, 98.77]
])
elec_cond_nh4cl_minus10 = 0.1*np.array([
    [10.68, 47.00, 69.78, 88.52],
    [8.184, 48.60, 70.18, 89.49],
    [10.36, 46.95, 68.30, 93.01]
])

elec_cond_mean_nh4cl_25, err_nh4cl_25 = mean_std(elec_cond_nh4cl_25)
elec_cond_mean_nh4cl_20, err_nh4cl_20 = mean_std(elec_cond_nh4cl_20)
elec_cond_mean_nh4cl_5, err_nh4cl_5 = mean_std(elec_cond_nh4cl_5)
elec_cond_mean_nh4cl_minus1, err_nh4cl_minus1 = mean_std(elec_cond_nh4cl_minus1)
elec_cond_mean_nh4cl_minus3, err_nh4cl_minus3 = mean_std(elec_cond_nh4cl_minus3)
elec_cond_mean_nh4cl_minus6, err_nh4cl_minus6 = mean_std(elec_cond_nh4cl_minus6)
elec_cond_mean_nh4cl_minus10, err_nh4cl_minus10 = mean_std(elec_cond_nh4cl_minus10)

err_nh4cl_25      = total_err(err_nh4cl_25,      elec_cond_mean_nh4cl_25)
err_nh4cl_20      = total_err(err_nh4cl_20,      elec_cond_mean_nh4cl_20)
err_nh4cl_5       = total_err(err_nh4cl_5,       elec_cond_mean_nh4cl_5)
err_nh4cl_minus1  = total_err(err_nh4cl_minus1,  elec_cond_mean_nh4cl_minus1)
err_nh4cl_minus3  = total_err(err_nh4cl_minus3,  elec_cond_mean_nh4cl_minus3)
err_nh4cl_minus6  = total_err(err_nh4cl_minus6,  elec_cond_mean_nh4cl_minus6)
err_nh4cl_minus10 = total_err(err_nh4cl_minus10, elec_cond_mean_nh4cl_minus10)

sigma_nh4cl_model = mc_sigma(
    conc_nh4cl, T_C_nh4cl,
    ionspec={"NH4_p1": 1.0, "Cl_m1": 1.0}
)

vars_exp_nh4cl = [
    elec_cond_mean_nh4cl_minus10, elec_cond_mean_nh4cl_minus6, elec_cond_mean_nh4cl_minus3,
    elec_cond_mean_nh4cl_minus1,  elec_cond_mean_nh4cl_5,      elec_cond_mean_nh4cl_20,
    elec_cond_mean_nh4cl_25
]
vars_mod_nh4cl = [sigma_nh4cl_model[:,i] for i in range(7)]
errs_nh4cl = [
    err_nh4cl_minus10, err_nh4cl_minus6, err_nh4cl_minus3,
    err_nh4cl_minus1,  err_nh4cl_5,      err_nh4cl_20,
    err_nh4cl_25
]
temp_labels_nh4cl = ['263.15 K','267.15 K','270.15 K','272.15 K','278.15 K','293.15 K','298.15 K']

# ==========================================
# Data: Na2CO3
# ==========================================
conc_na2co3 = np.array([5, 10, 25, 40, 55]) / 105.9888
# T_C_na2co3 = [-3, -1, 5, 20, 25]
T_C_na2co3 = [5, 20, 25]
tempsK_na2co3 = [t + 273.15 for t in T_C_na2co3]

elec_cond_na2co3_25 = 0.1*np.array([
    [8.1532, 14.718, 30.547, 44.517, 54.681],
    [8.1972, 14.685, 31.009, 43.703, 54.736],
    [8.3160, 14.564, 30.712, 43.439, 54.120]
])
elec_cond_na2co3_20 = 0.1*np.array([
    [7.412, 13.38, 27.77, 40.47, 49.71],
    [7.452, 13.35, 28.19, 39.73, 49.76],
    [7.560, 13.24, 27.92, 39.49, 49.20]
])
elec_cond_na2co3_5 = 0.1*np.array([
    [5.210, 9.306, 18.94, 26.91, 33.79],
    [5.123, 9.099, 19.00, 26.70, 33.97],
    [5.095, 9.031, 18.98, 26.70, 33.84]
])
elec_cond_na2co3_minus1 = 0.1*np.array([
    [6.495, 11.720, 24.336, 35.440, 43.561],
    [6.531, 11.690, 24.722, 34.785, 43.609],
    [6.624, 11.592, 24.460, 34.570, 43.043]
])
elec_cond_na2co3_minus3 = 0.1*np.array([
    [6.142, 11.088, 23.030, 33.570, 41.264],
    [6.176, 11.059, 23.392, 32.947, 41.310],
    [6.262, 10.966, 23.138, 32.741, 40.771]
])
elec_cond_na2co3_minus6 = 0.1*np.array([
    [3.971, 6.548, 13.99, 19.59, 24.34],
    [3.785, 6.521, 14.31, 19.81, 24.11],
    [3.602, 6.574, 13.87, 19.30, 24.09]
])
elec_cond_na2co3_minus10 = 0.1*np.array([
    [2.9648, 5.3520, 11.108, 16.188, 19.884],
    [2.9808, 5.3400, 11.276, 15.892, 19.904],
    [3.0240, 5.2960, 11.168, 15.796, 19.680]
])

elec_cond_mean_na2co3_25, err_na2co3_25 = mean_std(elec_cond_na2co3_25)
elec_cond_mean_na2co3_20, err_na2co3_20 = mean_std(elec_cond_na2co3_20)
elec_cond_mean_na2co3_5, err_na2co3_5 = mean_std(elec_cond_na2co3_5)
elec_cond_mean_na2co3_minus1, err_na2co3_minus1 = mean_std(elec_cond_na2co3_minus1)
elec_cond_mean_na2co3_minus3, err_na2co3_minus3 = mean_std(elec_cond_na2co3_minus3)
elec_cond_mean_na2co3_minus6, err_na2co3_minus6 = mean_std(elec_cond_na2co3_minus6)
elec_cond_mean_na2co3_minus10, err_na2co3_minus10 = mean_std(elec_cond_na2co3_minus10)

err_na2co3_25      = total_err(err_na2co3_25,      elec_cond_mean_na2co3_25)
err_na2co3_20      = total_err(err_na2co3_20,      elec_cond_mean_na2co3_20)
err_na2co3_5       = total_err(err_na2co3_5,       elec_cond_mean_na2co3_5)
err_na2co3_minus1  = total_err(err_na2co3_minus1,  elec_cond_mean_na2co3_minus1)
err_na2co3_minus3  = total_err(err_na2co3_minus3,  elec_cond_mean_na2co3_minus3)
err_na2co3_minus6  = total_err(err_na2co3_minus6,  elec_cond_mean_na2co3_minus6)
err_na2co3_minus10 = total_err(err_na2co3_minus10, elec_cond_mean_na2co3_minus10)

sigma_na2co3_model = mc_sigma(
    conc_na2co3, T_C_na2co3,
    ionspec={"Na_p1": 2.0, "CO3_m2": 1.0}
)

vars_exp_na2co3 = [
     elec_cond_mean_na2co3_5,      elec_cond_mean_na2co3_20,
    elec_cond_mean_na2co3_25
]
vars_mod_na2co3 = [sigma_na2co3_model[:,i] for i in range(3)]
errs_na2co3 = [
  err_na2co3_5,      err_na2co3_20,
    err_na2co3_25
]
temp_labels_na2co3 = ['278.15 K','293.15 K','298.15 K']

# ==========================================
# Plotting utilities per compound
# ==========================================
def plot_sigma_and_delta(conc, vars_exp, vars_mod, errs, temp_labels, title, limit=None,
                         gamry=None, gamry_err_frac=None, savebase=None):
    """Top: σ vs conc (molal) vs T; Bottom: Δ% vs conc (molal) vs T."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    colors = global_cmap

    # σ panels
    for j in range(len(errs)):
        safe_errorbar(ax1, conc, vars_exp[j], yerr=errs[j], color=colors[j], lw=LineWidth)
        safe_errorbar(ax1, conc, vars_mod[j], yerr=0.05, ls="--", color=colors[j], lw=LineWidth)

    # Optional Gamry
    if gamry is not None:
        gx, gy = gamry
        safe_errorbar(ax1, gx, gy, yerr=gy*gamry_err_frac, fmt='o', color='k',
                      mfc=colors[5], mec='k', ms=8)

    ax1.grid(True, alpha=0.6)
    ax1.set_ylabel(r'$\sigma$ (S/m)', fontsize=AxLabelSize)
    ax1.set_title(title, fontsize=AxLabelSize)
    if limit is not None:
        ax1.axvline(limit, ls="--", color=(0.7, 0, 0), lw=2)
        ax1.text(limit, ax1.get_ylim()[1], "McCleskey limit", color=(0.7, 0, 0),
                 fontsize=13, fontweight="bold", ha="left", va="top")

    tight_xlim(ax1, conc, pad_left=0.03, pad_right=0.03)

    ys_right = [np.asarray(vars_exp[j], float)[-1] if len(vars_exp[j])>0 else np.nan
                for j in range(len(vars_exp))]
    annotate_right(ax1, ys_right, temp_labels, fontsize=TLabelSize)

    # Δ panels
    diffs = [pdiff(vars_exp[j], vars_mod[j]) for j in range(len(vars_exp))]
    for j in range(len(vars_exp)):
        safe_errorbar(ax2, conc, diffs[j], yerr=5.0, color=colors[j], lw=LineWidth)

    # Optional Gamry Δ
    if gamry is not None:
        gx, gy = gamry
        # compare to interpolation of 20C model/exp if present; safer: interp to exp-20C if that was done
        # Here, we'll compare to exp at nearest temperature ~20 C if available in vars_exp order index 5.
        if len(vars_exp) >= 6:
            exp20 = np.asarray(vars_exp[5], float)
            probe = np.interp(gx, conc, exp20)
            diffg = pdiff(gy, probe)
            safe_errorbar(ax2, gx, diffg, yerr=5.0, fmt='o', color='k',
                          mfc=colors[5], mec='k', ms=8)

    ax2.grid(True, alpha=0.6)
    ax2.set_xlabel(r'Concentration (mol/kg$_{H2O}$)', fontsize=AxLabelSize)
    ax2.set_ylabel(r'$\Delta$ (%)', fontsize=AxLabelSize)
    tight_xlim(ax2, conc, pad_left=0.03, pad_right=0.03)

    plt.tight_layout()
    if SAVE_FIGS and savebase is not None:
        # fig.savefig(f"{savebase}.png", dpi=300)
        fig.savefig(f"{savebase}.pdf")

def plot_temperature_panels(tempsK, conc, vars_exp, vars_mod, errs, title,
                            gamry=None, gamry_err_frac=None, gamry_T_C=None, gamry_T_K=None,
                            model_at=None, savebase=None, freeze_skip=None):
    """
    Build σ(T) and Δ(T) for each concentration.
    Parameters
    ----------
    gamry : tuple or None
        (g_conc, g_sigma) arrays. Concentration must be in same units as `conc` (mol/kg_H2O).
        Sigma in S/m.
    gamry_err_frac : float or None
        Fractional uncertainty (e.g., 0.05 for 5%). Used as yerr = g_sigma * gamry_err_frac.
    gamry_T_C / gamry_T_K : float or None
        Temperature of Gamry measurement. Provide one.
    model_at : callable or None
        If provided, should be model_at(conc_array, T_K) -> sigma_model array (S/m),
        used to compute and plot Gamry Δ% on lower panel.
        If None, Gamry points are only shown on σ(T) panel.
    freeze_skip: slice or list of indices to skip (e.g., skip freezing region).
    """
    colors = global_cmap
    Mt = np.vstack(vars_exp)   # 7 × Nconc
    Mm = np.vstack(vars_mod)   # 7 × Nconc
    Me = np.vstack(errs)       # 7 × Nconc
    Md = np.vstack([pdiff(vars_exp[j], vars_mod[j]) for j in range(len(vars_exp))])

    concVT_exp = [Mt[:, i] for i in range(Mt.shape[1])]
    concVT_mod = [Mm[:, i] for i in range(Mm.shape[1])]
    errsVT     = [Me[:, i] for i in range(Me.shape[1])]
    diffsVT    = [Md[:, i] for i in range(Md.shape[1])]

    figT, (axT1, axT2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)

    idxs = range(len(conc))
    for i in idxs:
        e = concVT_exp[i]
        m = concVT_mod[i]
        ee = errsVT[i]
        if freeze_skip is not None:
            e  = np.asarray(e)[freeze_skip]
            m  = np.asarray(m)[freeze_skip]
            ee = np.asarray(ee)[freeze_skip]
            tk = np.asarray(tempsK)[freeze_skip]
        else:
            tk = tempsK

        safe_errorbar(axT1, tk, e, yerr=ee, color=colors[min(i, 6)], lw=LineWidth)
        safe_errorbar(axT1, tk, m, yerr=0.05, ls="--", color=colors[min(i, 6)], lw=LineWidth)

    axT1.grid(True, alpha=0.6)
    axT1.set_title(title, fontsize=AxLabelSize)
    axT1.set_ylabel(r'$\sigma$ (S/m)', fontsize=AxLabelSize)

    tmin, tmax = np.min(tempsK), np.max(tempsK)
    axT1.set_xlim(tmin, tmax + 0.02 * (tmax - tmin))  # 2% extra on the right

    # right-edge labels by concentration
    conc_labels = [f"{c:0.4f}" for c in conc]
    ys_right = [np.asarray(concVT_exp[i], float)[-1] for i in idxs]
    annotate_right(axT1, ys_right, conc_labels, fontsize=TLabelSize)

    for i in idxs:
        d = diffsVT[i]
        if freeze_skip is not None:
            d  = np.asarray(d)[freeze_skip]
            tk = np.asarray(tempsK)[freeze_skip]
        else:
            tk = tempsK
        safe_errorbar(axT2, tk, d, yerr=5.0, color=colors[min(i, 6)], lw=LineWidth)

    axT2.grid(True, alpha=0.6)
    axT2.set_xlabel('Temperature (K)', fontsize=AxLabelSize)
    axT2.set_ylabel(r'$\Delta$ (%)', fontsize=AxLabelSize)
    axT2.set_xlim(tmin, tmax + 0.02 * (tmax - tmin))

    # ------------------------------------------------------------------
    # Optional: overlay Gamry points at one temperature
    # ------------------------------------------------------------------
    # if gamry is not None: %turned off for now
    if False:
        g_conc, g_sig = gamry
        g_conc = np.asarray(g_conc, float).ravel()
        g_sig = np.asarray(g_sig, float).ravel()
        n = min(g_conc.size, g_sig.size)
        g_conc, g_sig = g_conc[:n], g_sig[:n]

        if gamry_T_K is None:
            if gamry_T_C is None:
                raise ValueError("Provide gamry_T_C or gamry_T_K when gamry is not None.")
            gamry_T_K = float(gamry_T_C) + 273.15
        else:
            gamry_T_K = float(gamry_T_K)

        # yerr for Gamry points
        if gamry_err_frac is None:
            g_yerr = None
        else:
            g_yerr = g_sig * float(gamry_err_frac)

        # Color each Gamry point by nearest concentration curve
        for gc, gs, gye in zip(g_conc, g_sig, (g_yerr if g_yerr is not None else np.full(n, np.nan))):
            i_near = int(np.nanargmin(np.abs(np.asarray(conc, float) - gc)))
            cidx = min(i_near, len(colors) - 1)
            # σ panel: point + optional errorbar
            safe_errorbar(axT1, [gamry_T_K], [gs],
                          yerr=None if g_yerr is None else [gye],
                          fmt='o', color='k', mec='k', mfc=colors[cidx], ms=8, lw=1.0)

        # Δ panel: only if model_at is provided
        if model_at is not None:
            g_model = np.asarray(model_at(g_conc, gamry_T_K), float).ravel()[:n]
            g_delta = pdiff(g_sig, g_model)
            for gc, gd in zip(g_conc, g_delta):
                i_near = int(np.nanargmin(np.abs(np.asarray(conc, float) - gc)))
                cidx = min(i_near, len(colors) - 1)
                safe_errorbar(axT2, [gamry_T_K], [gd],
                              yerr=5.0, fmt='o', color='k', mec='k', mfc=colors[cidx], ms=8, lw=1.0)

    plt.tight_layout()
    if SAVE_FIGS and savebase is not None:
        # figT.savefig(f"{savebase}.png", dpi=300)
        figT.savefig(f"{savebase}.pdf")

# ==========================================
# PLOTS: NaCl
# ==========================================
plot_sigma_and_delta(
    conc=conc_nacl,
    vars_exp=vars_exp_nacl,
    vars_mod=vars_mod_nacl,
    errs=errs_nacl,
    temp_labels=['263.15 K','267.15 K','270.15 K','272.15 K','278.15 K','293.15 K','298.15 K'],
    title='NaCl',
    limit=MC12NaCl_limit,
    gamry=(conc_gamNaCl, elec_gamNaCl),
    gamry_err_frac=errGamNaCl_scalar,
    savebase="graphique_nacl"
)
def nacl_model_at(conc_array, T_K, P_MPa=0.1):
    conc_array = np.asarray(conc_array, float)
    T_C = float(T_K) - 273.15
    ions = {"Na_p1": {"mols": conc_array}, "Cl_m1": {"mols": conc_array}}
    Tvec = np.full_like(conc_array, T_C)
    return elecCondMcCleskey2012(P_MPa, Tvec, ions)

# Temperature panels
plot_temperature_panels(
    tempsK=tempsK_nacl, conc=conc_nacl,
    vars_exp=vars_exp_nacl, vars_mod=vars_mod_nacl, errs=errs_nacl,
    title='NaCl (σ vs T and Δ vs T)', savebase="graphique_nacl_T",
    gamry=(conc_gamNaCl, elec_gamNaCl),
    gamry_err_frac=errGamNaCl_scalar,
    gamry_T_C=20.0, model_at=nacl_model_at
)

# ==========================================
# PLOTS: MgSO4
# ==========================================
plot_sigma_and_delta(
    conc=conc_mgso4,
    vars_exp=vars_exp_mgso4,
    vars_mod=vars_mod_mgso4,
    errs=errs_mgso4,
    temp_labels=temp_labels_mgso4,
    title='MgSO$_4$',
    limit=MC12MgSO4_limit,
    gamry=(conc_gamMgSO4, elec_gamMgSO4),
    gamry_err_frac=errGamMgSO4_scalar,
    savebase="graphique_mgso4"
)

def mgso4_model_at(conc_array, T_K, P_MPa=0.1):
    conc_array = np.asarray(conc_array, float)
    T_C = float(T_K) - 273.15
    ions = {"Mg_p2": {"mols": conc_array}, "SO4_m2": {"mols": conc_array}}
    Tvec = np.full_like(conc_array, T_C)
    return elecCondMcCleskey2012(P_MPa, Tvec, ions)

# Temperature panels (no special skip unless you want to omit freezing indices)
plot_temperature_panels(
    tempsK=tempsK_mgso4, conc=conc_mgso4,
    vars_exp=vars_exp_mgso4, vars_mod=vars_mod_mgso4, errs=errs_mgso4,
    title='MgSO$_4$ (σ vs T and Δ vs T)', savebase="graphique_mgso4_T",
    gamry=(conc_gamMgSO4, elec_gamMgSO4),
    gamry_err_frac=errGamMgSO4_scalar,
    gamry_T_C=19.5, model_at=nacl_model_at
)

# ==========================================
# PLOTS: NH4Cl
# ==========================================
plot_sigma_and_delta(
    conc=conc_nh4cl,
    vars_exp=vars_exp_nh4cl,
    vars_mod=vars_mod_nh4cl,
    errs=errs_nh4cl,
    temp_labels=temp_labels_nh4cl,
    title='NH$_4$Cl',
    limit=MC12NH4Cl_limit,
    savebase="graphique_nh4cl"
)
plot_temperature_panels(
    tempsK=tempsK_nh4cl, conc=conc_nh4cl,
    vars_exp=vars_exp_nh4cl, vars_mod=vars_mod_nh4cl, errs=errs_nh4cl,
    title='NH$_4$Cl (σ vs T and Δ vs T)', savebase="graphique_nh4cl_T"
)

# ==========================================
# PLOTS: Na2CO3
# ==========================================
# If you want to skip freezing-region isotherms in Na2CO3 T-panels, set freeze_skip = slice(4, None)
# (which corresponds to 5, 20, 25 C) – matches your MATLAB comment.
plot_sigma_and_delta(
    conc=conc_na2co3,
    vars_exp=vars_exp_na2co3,
    vars_mod=vars_mod_na2co3,
    errs=errs_na2co3,
    temp_labels=temp_labels_na2co3,
    title='Na$_2$CO$_3$',
    limit=MC12NaCO3_limit,
    savebase="graphique_na2co3"
)
plot_temperature_panels(
    tempsK=tempsK_na2co3, conc=conc_na2co3,
    vars_exp=vars_exp_na2co3, vars_mod=vars_mod_na2co3, errs=errs_na2co3,
    title='Na$_2$CO$_3$ (σ vs T and Δ vs T)',
    savebase="graphique_na2co3_T",
    freeze_skip=None  # option to keep only 5, 20, 25 C for T-panels, but we did this by just omitting the data from the array
)

# ==========================================
# Mixtures (1:1:1) — MgSO4 : NaCl : Na2CO3
# ==========================================
cmap = cm.get_cmap(chosen_map)
colors4 = cmap(np.arange(4))

conc_mixt = np.array([0.05, 0.1, 0.4, 0.7])
T_C_mixt = [-10, -1, 20, 25]
tempsK_mixt = [t + 273.15 for t in T_C_mixt]

# Model: σ for mixtures (sum of ionic contributions using given stoichiometry)
sigma_mixt_model = mc_sigma(
    conc_mixt, T_C_mixt,
    ionspec={
        "Na_p1": 3.0,
        "Cl_m1": 2.0,
        "Mg_p2": 1.0,
        "SO4_m2": 1.0,
        "CO3_m2": 1.0
    }
)  # shape 4 × 4 (T × conc)

# Experimental (mS/m → S/m)
elec_cond_mixt_25 = (np.array([
    [15.158, 28.017, 72.688, 62.458],
    [15.015, 26.730, 73.711, 66.418],
    [15.191, 26.389, 74.129, 65.230]
]) / 10.0)
elec_cond_mixt_20 = (np.array([
    [13.78, 25.47, 66.08, 56.78],
    [13.65, 24.30, 67.01, 60.38],
    [13.81, 23.99, 67.39, 59.30]
]) / 10.0)
elec_cond_mixt_minus1 = (np.array([
    [8.364, 15.24, 39.84, 35.69],
    [8.526, 15.76, 40.61, 35.17],
    [8.433, 14.68, 40.21, 36.40]
]) / 10.0)
elec_cond_mixt_minus10 = (np.array([
    [6.665, 11.81, 32.22, 31.56],
    [6.665, 10.90, 33.13, 28.42],
    [6.665, 11.355, 33.31, 32.63]
]) / 10.0)

elec_cond_mean_mixt_25, err_mixt_25 = mean_std(elec_cond_mixt_25)
elec_cond_mean_mixt_20, err_mixt_20 = mean_std(elec_cond_mixt_20)
elec_cond_mean_mixt_minus1, err_mixt_minus1 = mean_std(elec_cond_mixt_minus1)
elec_cond_mean_mixt_minus10, err_mixt_minus10 = mean_std(elec_cond_mixt_minus10)

err_mixt_25      = total_err(err_mixt_25,      elec_cond_mean_mixt_25)
err_mixt_20      = total_err(err_mixt_20,      elec_cond_mean_mixt_20)
err_mixt_minus1  = total_err(err_mixt_minus1,  elec_cond_mean_mixt_minus1)
err_mixt_minus10 = total_err(err_mixt_minus10, elec_cond_mean_mixt_minus10)

vars_exp_mixt = [
    elec_cond_mean_mixt_minus10, elec_cond_mean_mixt_minus1,
    elec_cond_mean_mixt_20,      elec_cond_mean_mixt_25
]
vars_mod_mixt = [sigma_mixt_model[:,i] for i in range(len(T_C_mixt))]
errs_mixt     = [err_mixt_minus10, err_mixt_minus1, err_mixt_20, err_mixt_25]
temp_labels_mixt = ['263.15 K','272.15 K','293.15 K','298.15 K']

# Plot mixtures σ and Δ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
for j in range(4):
    safe_errorbar(ax1, conc_mixt, vars_exp_mixt[j], yerr=errs_mixt[j],
                  color=colors4[j], lw=LineWidth)
    safe_errorbar(ax1, conc_mixt, vars_mod_mixt[j], yerr=0.05,
                  ls="--", color=colors4[j], lw=LineWidth)

ax1.grid(True, alpha=0.6)
ax1.set_title('MgSO$_4$:NaCl:Na$_2$CO$_3$ (1:1:1)', fontsize=AxLabelSize)
ax1.set_ylabel(r'$\sigma$ (S/m)', fontsize=AxLabelSize)
tight_xlim(ax1, conc_mixt, pad_left=0.03, pad_right=0.03)

ys_right = [np.asarray(vars_exp_mixt[j], float)[-1] for j in range(4)]
annotate_right(ax1, ys_right, temp_labels_mixt, fontsize=TLabelSize)

diffs_mixt = [pdiff(vars_exp_mixt[j], vars_mod_mixt[j]) for j in range(4)]
for j in range(4):
    safe_errorbar(ax2, conc_mixt, diffs_mixt[j], yerr=5.0,
                  color=colors4[j], lw=LineWidth)

ax2.grid(True, alpha=0.6)
ax2.set_xlabel(r'Concentration (mol/kg$_{H2O}$)', fontsize=AxLabelSize)
ax2.set_ylabel(r'$\Delta$ (%)', fontsize=AxLabelSize)
tight_xlim(ax2, conc_mixt, pad_left=0.03, pad_right=0.03)

plt.tight_layout()
if SAVE_FIGS:
    # fig.savefig("graphique_mixtures.png", dpi=300)
    fig.savefig("graphique_mixtures.pdf")

# Mixtures temperature panels
# Build (T, conc) matrices
Mt = np.vstack(vars_exp_mixt)   # 4 × 4
Mm = np.vstack(vars_mod_mixt)   # 4 × 4
Me = np.vstack(errs_mixt)       # 4 × 4
Md = np.vstack(diffs_mixt)      # 4 × 4

concVT_exp = [Mt[:, i] for i in range(Mt.shape[1])]
concVT_mod = [Mm[:, i] for i in range(Mm.shape[1])]
errsVT     = [Me[:, i] for i in range(Me.shape[1])]
diffsVT    = [Md[:, i] for i in range(Md.shape[1])]

figT, (axT1, axT2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
for i in range(len(conc_mixt)):
    safe_errorbar(axT1, tempsK_mixt, concVT_exp[i], yerr=errsVT[i],
                  color=colors4[i], lw=LineWidth)
    safe_errorbar(axT1, tempsK_mixt, concVT_mod[i], yerr=0.05,
                  ls="--", color=colors4[i], lw=LineWidth)

axT1.grid(True, alpha=0.6)
axT1.set_title('MgSO$_4$:NaCl:Na$_2$CO$_3$ (1:1:1)', fontsize=AxLabelSize)
axT1.set_ylabel(r'$\sigma$ (S/m)', fontsize=AxLabelSize)

conc_labels = [f"{c:0.4f}" for c in conc_mixt]
ys_right = [np.asarray(concVT_exp[i], float)[-1] for i in range(len(conc_mixt))]
annotate_right(axT1, ys_right, conc_labels, fontsize=TLabelSize)

for i in range(len(conc_mixt)):
    safe_errorbar(axT2, tempsK_mixt, diffsVT[i], yerr=5.0,
                  color=colors4[i], lw=LineWidth)

axT2.grid(True, alpha=0.6)
axT2.set_xlabel('Temperature (K)', fontsize=AxLabelSize)
axT2.set_ylabel(r'$\Delta$ (%)', fontsize=AxLabelSize)

plt.tight_layout()
if SAVE_FIGS:
    # figT.savefig("graphique_mixtures_T.png", dpi=300)
    figT.savefig("graphique_mixtures_T.pdf")

# Show figures if running interactively
plt.show()
