#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiPOZ Plotting Module

Comprehensive plotting functions for:
- Gamry impedance data (Nyquist, Bode, timeseries)
- Conductivity vs concentration
- Conductivity vs temperature
- Conductivity vs pressure
- External benchtop probe data

Adapted from gamryPlots.py and MahboubEtAl2026.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import logging

# Configure logger
log = logging.getLogger('HiPOZ')

# ========================================
# Matplotlib Configuration
# ========================================
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["font.size"] = 14
plt.rcParams["pdf.fonttype"] = 42   # embed as TrueType (editable in Illustrator)
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

# Try LaTeX if available (for publication-quality plots)
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{stix}\usepackage{siunitx}\usepackage{upgreek}\usepackage[version=4]{mhchem}\sisetup{round-mode=places,scientific-notation=true,round-precision=2}'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'STIXGeneral'
except:
    log.warning("LaTeX not available for plotting. Using standard matplotlib text rendering.")

# Plot styling
AxLabelSize = 18
TLabelSize = 10
LineWidth = 2
MS_data = 'o'
LS_fit = '-'

# ========================================
# Helper Functions
# ========================================

def safe_errorbar(ax, x, y, yerr=None, **kwargs):
    """
    Trim x/y to common length and handle mismatched yerr to avoid size errors.

    Parameters
    ----------
    ax : matplotlib axis
    x, y : array-like
        Data coordinates
    yerr : array-like or scalar, optional
        Error bars
    **kwargs : additional arguments passed to ax.errorbar
    """
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
        # Default to 5% if yerr doesn't match
        yerr_use = 0.05 * (np.nanmean(np.abs(y)) if np.isfinite(y).any() else 1.0)

    return ax.errorbar(x, y, yerr=yerr_use, **kwargs)


def pdiff(x, y):
    """
    Percent difference: 100*(x - y)/y with NaN-safe behavior.

    Parameters
    ----------
    x, y : array-like
        Values to compare

    Returns
    -------
    out : ndarray
        Percent difference
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y != 0)
    out[mask] = 100.0 * (x[mask] - y[mask]) / y[mask]
    return out


def tight_xlim(ax, data, pad=0.03, pad_left=None, pad_right=None):
    """
    Set xlim snug to data with optional asymmetric padding.

    Parameters
    ----------
    ax : matplotlib axis
    data : array-like
        Data to fit
    pad : float
        Symmetric fraction if left/right not given
    pad_left, pad_right : float, optional
        Fractions of span added to each side
    """
    c = np.asarray(data, dtype=float).ravel()
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


def annotate_right(ax, ys_raw, labels, *,
                   fontsize=10,
                   x_pad_frac=0.08,
                   x_text_rel=1.005,
                   x_conn_rel=1.03,
                   min_px_gap=24, n_iter=60, k_push=0.25,
                   connector=True):
    """
    Place right-edge labels with non-overlap and auto xlim padding.

    Parameters
    ----------
    ax : matplotlib axis
    ys_raw : array-like
        Y-coordinates for labels (typically final values of curves)
    labels : list of str
        Label text for each curve
    fontsize : int
        Font size for labels
    x_pad_frac : float
        Fraction of x-span to add on right for labels
    x_text_rel : float
        Relative position for text (0-1 from left to right)
    x_conn_rel : float
        Relative position for connector end
    min_px_gap : int
        Minimum vertical gap between labels (pixels)
    n_iter : int
        Iterations for label repulsion algorithm
    k_push : float
        Repulsion strength
    connector : bool
        Whether to draw connector lines
    """
    # Find rightmost plotted x
    x_end = None
    for line in ax.lines:
        xd = np.asarray(line.get_xdata(), float)
        xd = xd[np.isfinite(xd)]
        if xd.size:
            x_end = xd.max() if x_end is None else max(x_end, xd.max())

    if x_end is None:
        x_end = ax.get_xlim()[1]

    # Extend xlim
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

    # Pixel → data spacing
    inv = ax.transData.inverted()
    y0_pix = ax.transData.transform((0, ax.get_ylim()[0]))[1]
    min_dy = abs(inv.transform((0, y0_pix + min_px_gap))[1] - ax.get_ylim()[0])

    # Sort and repulse labels
    order = np.argsort(ys[idx])
    ywork = ys[idx][order].copy()

    for _ in range(n_iter):
        moved = False
        for j in range(1, ywork.size):
            gap = ywork[j] - ywork[j-1]
            if gap < min_dy:
                push = (min_dy - gap) * k_push
                ywork[j] += push
                ywork[j-1] -= push
                moved = True
        if not moved:
            break

    ys_out = ys.copy()
    ys_out[idx[order]] = ywork

    # Draw connectors and text
    for y0, y1, s in zip(ys, ys_out, labels):
        if not np.isfinite(y1):
            continue
        if connector and np.isfinite(y0):
            ax.plot([x_end, x_conn], [y0, y1],
                    lw=1.8, ls=":", color="0.4", zorder=2.5)
        ax.text(x_text, y1, s, ha="left", va="center",
                fontsize=fontsize, clip_on=False, zorder=3)


# ========================================
# Conductivity vs Concentration Plotting
# ========================================

def plot_sigma_vs_concentration(conc_data, sigma_data, temp_labels=None,
                                sigma_errors=None, model_data=None,
                                model_data_corrected=None,
                                xlabel=r'Concentration (mol/kg$_{\mathrm{H_2O}}$)',
                                title='Conductivity vs Concentration',
                                show_delta=False, show_title=True, limit=None,
                                out_file=None, colormap='tab10',
                                fontsize_label=None, fontsize_title=None,
                                fontsize_legend=None):
    """
    Plot conductivity vs concentration with multiple temperature curves.

    Parameters
    ----------
    conc_data : array-like (N,)
        Concentration values (e.g., molal)
    sigma_data : list of array-like, length M
        Each element is conductivity array at one temperature (length N)
    temp_labels : list of str, length M, optional
        Temperature labels (e.g., ['263.15 K', '273.15 K', ...])
    sigma_errors : list of array-like, length M, optional
        Error bars for each temperature curve
    model_data : list of array-like, length M, optional
        Model predictions for comparison (plotted as dashed lines)
    model_data_corrected : list of array-like, length M, optional
        Second model series, plotted dotted in the same per-series colours.
        Intended for the WATEQ4F-speciated McCleskey model alongside the
        unspeciated one. When both are given the two curves are labelled
        'MC12' and 'MC12 + WATEQ4F'.
    xlabel : str
        X-axis label
    title : str
        Plot title
    show_delta : bool
        If True and model_data provided, show percent difference panel below
    limit : float, optional
        Vertical line marking validity limit
    out_file : str, optional
        Save figure to this file
    colormap : str
        Matplotlib colormap name
    fontsize_label : int, optional
        Font size for axis labels (default: AxLabelSize global)
    fontsize_title : int, optional
        Font size for title (default: AxLabelSize global)
    fontsize_legend : int, optional
        Font size for legend (default: TLabelSize global)
    """
    # Set font sizes to defaults if not specified
    if fontsize_label is None:
        fontsize_label = AxLabelSize
    if fontsize_title is None:
        fontsize_title = AxLabelSize
    if fontsize_legend is None:
        fontsize_legend = TLabelSize
    conc = np.asarray(conc_data, dtype=float)
    n_temps = len(sigma_data)

    if temp_labels is None:
        temp_labels = [f'T{i+1}' for i in range(n_temps)]

    if sigma_errors is None:
        sigma_errors = [None] * n_temps

    # Setup figure
    if show_delta and model_data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None

    # Get colors - use discrete indices for qualitative colormaps like tab10
    cmap = cm.get_cmap(colormap)
    if 'tab' in colormap or 'Set' in colormap:  # Discrete colormap
        colors = cmap(np.arange(n_temps) % 10)
    else:  # Continuous colormap (plasma, viridis, etc.)
        colors = cmap(np.linspace(0, 1, n_temps))

    # Plot experimental data (markers only, no lines)
    for i, (sigma, err, label) in enumerate(zip(sigma_data, sigma_errors, temp_labels)):
        safe_errorbar(ax1, conc, sigma, yerr=err, fmt='o',
                     color=colors[i], ms=8, mew=1.5, capsize=4, label=label)

    # Plot model data if provided (dashed lines for McCleskey model).
    # Only label the model curves when both are drawn and so need telling apart;
    # labelling unconditionally would insert a new "MC12" entry into existing
    # legends (study_plots passes show_legend through to ax.legend()).
    label_models = model_data is not None and model_data_corrected is not None
    if model_data is not None:
        for i, sigma_model in enumerate(model_data):
            ax1.plot(conc, sigma_model, ls="--", color=colors[i], lw=LineWidth,
                    alpha=0.7, zorder=1,
                    label='MC12' if (label_models and i == 0) else None)

    # Speciated model (dotted), same per-series colours so the pair reads together
    if model_data_corrected is not None:
        for i, sigma_model in enumerate(model_data_corrected):
            ax1.plot(conc, sigma_model, ls=":", color=colors[i], lw=LineWidth,
                    alpha=0.9, zorder=1.5,
                    label='MC12 + WATEQ4F' if i == 0 else None)

    # Formatting
    ax1.grid(True, alpha=0.6)
    ax1.set_ylabel(r'$\sigma$ (S/m)', fontsize=fontsize_label)
    if show_title:
        ax1.set_title(title, fontsize=fontsize_title)

    # Set y-axis minimum to 0 (conductivity cannot be negative)
    ax1.set_ylim(bottom=0)

    # Draw McCleskey applicability limit if specified
    if limit is not None:
        ax1.axvline(limit, ls="--", color=(0.8, 0, 0), lw=2.5, alpha=0.8, zorder=2.5)
        # Position text label at 95% of y-axis height
        y_pos = ax1.get_ylim()[0] + 0.95 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        # This ceiling is the validity range of the *unspeciated* model: for a
        # 2:2 electrolyte such as MgSO4 it is very low (0.01245 mol/kg) because
        # ion association is ignored. The speciated curve remains usable well
        # beyond it, so name which curve the limit applies to when both are shown.
        limit_label = ("McCleskey limit\n({:.4f} mol/kg)".format(limit)
                       if model_data_corrected is None else
                       "MC12 limit\n({:.4f} mol/kg)".format(limit))
        ax1.text(limit * 1.01, y_pos, limit_label,
                color=(0.8, 0, 0), fontsize=11, fontweight="bold",
                ha="left", va="top", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", edgecolor=(0.8, 0, 0), alpha=0.8))

    tight_xlim(ax1, conc, pad_left=0.03, pad_right=0.10)

    # Right-edge labels
    ys_right = [np.asarray(sigma, float)[-1] if len(sigma) > 0 else np.nan
                for sigma in sigma_data]
    annotate_right(ax1, ys_right, temp_labels, fontsize=fontsize_legend)

    # Delta panel (percent difference from model)
    if show_delta and model_data is not None and ax2 is not None:
        for i, (sigma_exp, sigma_mod) in enumerate(zip(sigma_data, model_data)):
            delta = pdiff(sigma_exp, sigma_mod)
            safe_errorbar(ax2, conc, delta, yerr=5.0, fmt='o',
                         color=colors[i], ms=6, mew=1.2, capsize=3)

        ax2.grid(True, alpha=0.6)
        ax2.set_xlabel(xlabel, fontsize=fontsize_label)
        ax2.set_ylabel(r'$\Delta$ (\%)', fontsize=fontsize_label)
        tight_xlim(ax2, conc, pad_left=0.03, pad_right=0.10)
    else:
        ax1.set_xlabel(xlabel, fontsize=fontsize_label)

    plt.tight_layout()

    if out_file:
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        log.info(f'Sigma vs concentration plot saved: {out_file}')

    return fig


# ========================================
# Conductivity vs Temperature Plotting
# ========================================

def plot_sigma_vs_temperature(temp_data, sigma_data, conc_labels=None,
                              sigma_errors=None, model_data=None,
                              model_data_corrected=None,
                              xlabel=r'Temperature (K)',
                              title='Conductivity vs Temperature',
                              show_delta=False, show_title=True, out_file=None,
                              colormap='tab10',
                              fontsize_label=None, fontsize_title=None,
                              fontsize_legend=None):
    """
    Plot conductivity vs temperature with multiple concentration curves.

    Parameters
    ----------
    temp_data : array-like (N,)
        Temperature values (K)
    sigma_data : list of array-like, length M
        Each element is conductivity array at one concentration (length N)
    conc_labels : list of str, length M, optional
        Concentration labels (e.g., ['0.5 mol/kg', '1.0 mol/kg', ...])
    sigma_errors : list of array-like, length M, optional
        Error bars for each concentration curve
    model_data : list of array-like, length M, optional
        Model predictions for comparison (plotted as dashed lines)
    model_data_corrected : list of array-like, length M, optional
        Second model series, plotted dotted in the same per-series colours.
        Intended for the WATEQ4F-speciated McCleskey model alongside the
        unspeciated one. When both are given the two curves are labelled
        'MC12' and 'MC12 + WATEQ4F'.
    xlabel : str
        X-axis label
    title : str
        Plot title
    show_delta : bool
        If True and model_data provided, show percent difference panel below
    out_file : str, optional
        Save figure to this file
    colormap : str
        Matplotlib colormap name
    fontsize_label : int, optional
        Font size for axis labels (default: AxLabelSize global)
    fontsize_title : int, optional
        Font size for title (default: AxLabelSize global)
    fontsize_legend : int, optional
        Font size for legend (default: TLabelSize global)
    """
    # Set font sizes to defaults if not specified
    if fontsize_label is None:
        fontsize_label = AxLabelSize
    if fontsize_title is None:
        fontsize_title = AxLabelSize
    if fontsize_legend is None:
        fontsize_legend = TLabelSize
    temps = np.asarray(temp_data, dtype=float)
    n_concs = len(sigma_data)

    if conc_labels is None:
        conc_labels = [f'C{i+1}' for i in range(n_concs)]

    if sigma_errors is None:
        sigma_errors = [None] * n_concs

    # Setup figure
    if show_delta and model_data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None

    # Get colors - use discrete indices for qualitative colormaps like tab10
    cmap = cm.get_cmap(colormap)
    if 'tab' in colormap or 'Set' in colormap:  # Discrete colormap
        colors = cmap(np.arange(n_concs) % 10)
    else:  # Continuous colormap (plasma, viridis, etc.)
        colors = cmap(np.linspace(0, 1, n_concs))

    # Plot experimental data (markers only, no lines)
    for i, (sigma, err, label) in enumerate(zip(sigma_data, sigma_errors, conc_labels)):
        safe_errorbar(ax1, temps, sigma, yerr=err, fmt='o',
                     color=colors[i], ms=8, mew=1.5, capsize=4, label=label)

    # Plot model data if provided (dashed lines for McCleskey model).
    # See plot_sigma_vs_concentration for why labels are conditional.
    label_models = model_data is not None and model_data_corrected is not None
    if model_data is not None:
        for i, sigma_model in enumerate(model_data):
            ax1.plot(temps, sigma_model, ls="--", color=colors[i], lw=LineWidth,
                    alpha=0.7, zorder=1,
                    label='MC12' if (label_models and i == 0) else None)

    # Speciated model (dotted), same per-series colours
    if model_data_corrected is not None:
        for i, sigma_model in enumerate(model_data_corrected):
            ax1.plot(temps, sigma_model, ls=":", color=colors[i], lw=LineWidth,
                    alpha=0.9, zorder=1.5,
                    label='MC12 + WATEQ4F' if i == 0 else None)

    # Formatting
    ax1.grid(True, alpha=0.6)
    if show_title:
        ax1.set_title(title, fontsize=fontsize_title)
    ax1.set_ylabel(r'$\sigma$ (S/m)', fontsize=fontsize_label)

    # Set y-axis minimum to 0 (conductivity cannot be negative)
    ax1.set_ylim(bottom=0)

    tmin, tmax = np.min(temps), np.max(temps)
    ax1.set_xlim(tmin, tmax + 0.10 * (tmax - tmin))  # 10% extra on right

    # Right-edge labels
    ys_right = [np.asarray(sigma, float)[-1] for sigma in sigma_data]
    annotate_right(ax1, ys_right, conc_labels, fontsize=fontsize_legend)

    # Delta panel (percent difference from model)
    if show_delta and model_data is not None and ax2 is not None:
        for i, (sigma_exp, sigma_mod) in enumerate(zip(sigma_data, model_data)):
            delta = pdiff(sigma_exp, sigma_mod)
            safe_errorbar(ax2, temps, delta, yerr=5.0, fmt='o',
                         color=colors[i], ms=6, mew=1.2, capsize=3)

        ax2.grid(True, alpha=0.6)
        ax2.set_xlabel(xlabel, fontsize=fontsize_label)
        ax2.set_ylabel(r'$\Delta$ (\%)', fontsize=fontsize_label)
        ax2.set_xlim(tmin, tmax + 0.10 * (tmax - tmin))
    else:
        ax1.set_xlabel(xlabel, fontsize=fontsize_label)

    plt.tight_layout()

    if out_file:
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        log.info(f'Sigma vs temperature plot saved: {out_file}')

    return fig


# ========================================
# Gamry Data Plotting (from gamryPlots.py)
# ========================================

def plot_nyquist(solutions, out_file=None, title='Nyquist Plot',
                show_fit=True, colormap='viridis'):
    """
    Plot Nyquist diagram (Re(Z) vs -Im(Z)).

    Parameters
    ----------
    solutions : list
        List of Solution objects with Z_ohm, Zfit_ohm attributes
    out_file : str, optional
        Output file path
    title : str
        Plot title
    show_fit : bool
        Whether to show circuit fit
    colormap : str
        Matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(solutions)))

    for i, sol in enumerate(solutions):
        label = getattr(sol, 'legLabel', f'Sol {i+1}')
        color = colors[i]

        # Plot data
        ax.scatter(np.real(sol.Z_ohm), -np.imag(sol.Z_ohm),
                  marker='o', label=f'{label} data',
                  color=color, alpha=0.6)

        # Plot fit
        if show_fit and hasattr(sol, 'Zfit_ohm') and sol.Zfit_ohm is not None:
            ax.plot(np.real(sol.Zfit_ohm), -np.imag(sol.Zfit_ohm),
                   ls='-', label=f'{label} fit', color=color)

    ax.set_xlabel(r'$\mathrm{Re}\{Z\}$ ($\Omega$)', fontsize=AxLabelSize)
    ax.set_ylabel(r'$-\mathrm{Im}\{Z\}$ ($\Omega$)', fontsize=AxLabelSize)
    ax.set_title(title, fontsize=AxLabelSize)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()

    if out_file:
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        log.info(f'Nyquist plot saved: {out_file}')

    plt.close()


def plot_bode(solutions, out_file=None, title='Bode Plot',
             show_fit=True, colormap='viridis'):
    """
    Plot Bode diagram (|Z| and phase vs frequency).

    Parameters
    ----------
    solutions : list
        List of Solution objects with f_Hz, Z_ohm attributes
    out_file : str, optional
        Output file path
    title : str
        Plot title
    show_fit : bool
        Whether to show circuit fit
    colormap : str
        Matplotlib colormap name
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    cmap = cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(solutions)))

    for i, sol in enumerate(solutions):
        label = getattr(sol, 'legLabel', f'Sol {i+1}')
        color = colors[i]

        # Magnitude
        ax1.scatter(sol.f_Hz, np.abs(sol.Z_ohm),
                   marker='o', label=f'{label} data',
                   color=color, alpha=0.6)
        if show_fit and hasattr(sol, 'Zfit_ohm') and sol.Zfit_ohm is not None:
            ax1.plot(sol.f_Hz, np.abs(sol.Zfit_ohm),
                    ls='-', color=color)

        # Phase
        ax2.scatter(sol.f_Hz, np.angle(sol.Z_ohm, deg=True),
                   marker='o', color=color, alpha=0.6)
        if show_fit and hasattr(sol, 'Zfit_ohm') and sol.Zfit_ohm is not None:
            ax2.plot(sol.f_Hz, np.angle(sol.Zfit_ohm, deg=True),
                    ls='-', color=color)

    ax1.set_ylabel(r'$|Z|$ ($\Omega$)', fontsize=AxLabelSize)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(title, fontsize=AxLabelSize)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel(r'Frequency $f$ (Hz)', fontsize=AxLabelSize)
    ax2.set_ylabel(r'Phase ($^\circ$)', fontsize=AxLabelSize)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_file:
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        log.info(f'Bode plot saved: {out_file}')

    plt.close()


def plot_conductivity_vs_pressure(measurements, out_file=None,
                                  title='Conductivity vs Pressure',
                                  color_by='temperature'):
    """
    Plot conductivity vs pressure, colored by temperature or concentration.

    Parameters
    ----------
    measurements : list
        List of Solution objects with P_MPa, sigma_Sm, T_K attributes
    out_file : str, optional
        Output file path
    title : str
        Plot title
    color_by : str
        'temperature' or 'concentration' - what to use for color coding
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    P = np.array([m.P_MPa for m in measurements if hasattr(m, 'P_MPa') and hasattr(m, 'sigma_Sm')])
    sigma = np.array([m.sigma_Sm for m in measurements if hasattr(m, 'P_MPa') and hasattr(m, 'sigma_Sm')])

    if color_by == 'temperature':
        T = np.array([m.T_K for m in measurements if hasattr(m, 'P_MPa') and hasattr(m, 'sigma_Sm') and hasattr(m, 'T_K')])
        scatter = ax.scatter(P, sigma, c=T, cmap='plasma', s=50, edgecolors='k')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (K)', fontsize=14)
    else:
        ax.plot(P, sigma, 'o', markersize=8, markerfacecolor='g', markeredgecolor='k')

    ax.set_xlabel('Pressure $P$ (MPa)', fontsize=AxLabelSize)
    ax.set_ylabel('Conductivity $\sigma$ (S/m)', fontsize=AxLabelSize)
    ax.set_title(title, fontsize=AxLabelSize)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_file:
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        log.info(f'Conductivity vs pressure plot saved: {out_file}')

    plt.close()


# ========================================
# Data Loading Utilities
# ========================================

def load_external_data(file_path, format='csv'):
    """
    Load external conductivity data from file.

    Parameters
    ----------
    file_path : str
        Path to data file
    format : str
        'csv', 'txt', or 'npy'

    Returns
    -------
    data : dict
        Dictionary with keys: 'concentration', 'temperature', 'conductivity',
        'errors' (optional)
    """
    if format == 'csv':
        import csv
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            data = {'concentration': [], 'temperature': [],
                   'conductivity': [], 'errors': []}
            for row in reader:
                data['concentration'].append(float(row.get('concentration', row.get('conc', 0))))
                data['temperature'].append(float(row.get('temperature', row.get('T', 0))))
                data['conductivity'].append(float(row.get('conductivity', row.get('sigma', 0))))
                data['errors'].append(float(row.get('error', row.get('unc', 0))))

            for key in data:
                data[key] = np.array(data[key])

    elif format == 'npy':
        data = np.load(file_path, allow_pickle=True).item()

    else:  # txt
        arr = np.loadtxt(file_path)
        data = {
            'concentration': arr[:, 0],
            'temperature': arr[:, 1],
            'conductivity': arr[:, 2],
            'errors': arr[:, 3] if arr.shape[1] > 3 else None
        }

    return data


def group_data_by_variable(concentrations, temperatures, conductivities,
                          group_by='temperature', aggregate_func=np.mean):
    """
    Group conductivity data by temperature or concentration for plotting.

    This is a general function that can work with any tabular conductivity dataset.

    Parameters
    ----------
    concentrations : array-like
        Concentration values (any units)
    temperatures : array-like
        Temperature values (any units)
    conductivities : array-like
        Conductivity values (S/m)
    group_by : str
        'temperature' - group by temperature for σ vs concentration plots
        'concentration' - group by concentration for σ vs temperature plots
    aggregate_func : callable
        Function to aggregate multiple measurements (default: np.mean)

    Returns
    -------
    x_values : ndarray
        Unique x-axis values (conc or temp depending on group_by)
    y_by_group : list of ndarray
        Conductivity arrays for each group
    labels : list of str
        Labels for each group
    """
    concentrations = np.asarray(concentrations)
    temperatures = np.asarray(temperatures)
    conductivities = np.asarray(conductivities)

    if group_by == 'temperature':
        # Group by temperature → plot σ vs concentration
        unique_temps = np.unique(temperatures)
        unique_concs = np.unique(concentrations)

        y_by_group = []
        labels = []

        for temp in unique_temps:
            labels.append(f'{temp:.2f}')

            sigma_at_temp = []
            for conc in unique_concs:
                mask = (concentrations == conc) & (temperatures == temp)
                sigma_subset = conductivities[mask]

                if len(sigma_subset) > 0:
                    sigma_at_temp.append(aggregate_func(sigma_subset))
                else:
                    sigma_at_temp.append(np.nan)

            y_by_group.append(np.array(sigma_at_temp))

        return unique_concs, y_by_group, labels

    elif group_by == 'concentration':
        # Group by concentration → plot σ vs temperature
        unique_temps = np.unique(temperatures)
        unique_concs = np.unique(concentrations)

        y_by_group = []
        labels = []

        for conc in unique_concs:
            labels.append(f'{conc:.4f}')

            sigma_at_conc = []
            for temp in unique_temps:
                mask = (concentrations == conc) & (temperatures == temp)
                sigma_subset = conductivities[mask]

                if len(sigma_subset) > 0:
                    sigma_at_conc.append(aggregate_func(sigma_subset))
                else:
                    sigma_at_conc.append(np.nan)

            y_by_group.append(np.array(sigma_at_conc))

        return unique_temps, y_by_group, labels

    else:
        raise ValueError("group_by must be 'temperature' or 'concentration'")
