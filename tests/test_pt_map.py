#!/usr/bin/env python
"""
Test the 2D sigma(P,T) map plotting used by the GUI's σ(P,T) tab.

Exercises plot_conductivity_PT directly with a plain matplotlib Figure so it
runs headless (Agg, no Qt, no LaTeX required).
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gamryPlots import plot_conductivity_PT


@pytest.fixture
def fig():
    # The gamryPlots import turns LaTeX rendering on; disable it here (restored
    # after each test) so these tests run without a TeX installation.
    was_usetex = plt.rcParams['text.usetex']
    plt.rcParams['text.usetex'] = False
    f = plt.figure()
    yield f
    plt.close(f)
    plt.rcParams['text.usetex'] = was_usetex


def make_grid_data(nP=5, nT=4):
    """Synthetic KCl-standard-like sweep: sigma rises with T, falls with P."""
    P, T = np.meshgrid(np.linspace(0, 400, nP), np.linspace(263, 323, nT))
    P = P.ravel()
    T = T.ravel()
    sigma = 8.0 + 0.05 * (T - 298.15) - 0.002 * P
    return P, T, sigma


def test_pt_map_full_grid(fig):
    P, T, sigma = make_grid_data()
    ax, sc = plot_conductivity_PT(fig, P, T, sigma)

    assert sc is not None, "Scatter artist should be returned for valid data"
    assert sc.get_offsets().shape[0] == len(P), "All points should be plotted"
    assert len(fig.axes) >= 2, "Colorbar axes should be present"
    assert ax.get_xlabel() == 'P (MPa)'
    assert 'T' in ax.get_ylabel()
    # Interpolated background should have been drawn for a spanning grid
    assert len(ax.collections) > 1, "Contour background expected in addition to scatter"

    fig.canvas.draw()  # exercise actual rendering


def test_pt_map_reference_contour(fig):
    P, T, sigma = make_grid_data()
    ax, sc = plot_conductivity_PT(fig, P, T, sigma, sigma_ref_Sm=8.0)
    assert sc is not None
    fig.canvas.draw()


def test_pt_map_drops_nonfinite(fig):
    P, T, sigma = make_grid_data()
    sigma = sigma.copy()
    sigma[::3] = np.nan
    ax, sc = plot_conductivity_PT(fig, P, T, sigma)
    n_expected = np.isfinite(sigma).sum()
    assert sc.get_offsets().shape[0] == n_expected


def test_pt_map_collinear_points(fig):
    # All at one pressure: triangulation impossible, scatter must still work
    T = np.linspace(263, 323, 6)
    P = np.full_like(T, 100.0)
    sigma = 8.0 + 0.05 * (T - 298.15)
    ax, sc = plot_conductivity_PT(fig, P, T, sigma)
    assert sc is not None
    assert sc.get_offsets().shape[0] == len(T)
    fig.canvas.draw()


def test_pt_map_constant_sigma(fig):
    P, T, _ = make_grid_data()
    sigma = np.full_like(P, 8.0)
    ax, sc = plot_conductivity_PT(fig, P, T, sigma)
    assert sc is not None
    fig.canvas.draw()


def test_pt_map_two_points(fig):
    ax, sc = plot_conductivity_PT(fig, [0.1, 200.0], [298.15, 298.15], [8.0, 7.5])
    assert sc is not None
    assert sc.get_offsets().shape[0] == 2


def test_pt_map_empty(fig):
    ax, sc = plot_conductivity_PT(fig, [], [], [])
    assert sc is None
    assert len(ax.texts) > 0, "Empty plot should show a message"
    fig.canvas.draw()


def test_pt_map_temperature_in_celsius(fig):
    # 298.15 K should land at 25 degC on the y axis
    ax, sc = plot_conductivity_PT(fig, [100.0, 200.0, 100.0, 200.0],
                                  [298.15, 298.15, 273.15, 273.15],
                                  [8.0, 7.8, 6.5, 6.3])
    ys = sc.get_offsets()[:, 1]
    assert np.isclose(ys.max(), 25.0)
    assert np.isclose(ys.min(), 0.0)


# --- 3D surface companion (plot_conductivity_PT_surface) ---------------------

from gamryPlots import plot_conductivity_PT_surface


def test_pt_surface_full_grid(fig):
    P, T, sigma = make_grid_data()
    ax, artist = plot_conductivity_PT_surface(fig, P, T, sigma, sigma_ref_Sm=8.0)
    assert ax.name == '3d', "Should produce 3D axes"
    assert artist is not None
    assert len(fig.axes) >= 2, "Colorbar axes should be present"
    assert ax.get_zlabel() != ""
    fig.canvas.draw()


def test_pt_surface_collinear_falls_back_to_scatter(fig):
    T = np.linspace(263, 323, 6)
    P = np.full_like(T, 100.0)
    sigma = 8.0 + 0.05 * (T - 298.15)
    ax, artist = plot_conductivity_PT_surface(fig, P, T, sigma)
    assert ax.name == '3d'
    assert artist is not None, "Scatter fallback expected for collinear points"
    fig.canvas.draw()


def test_pt_surface_empty(fig):
    ax, artist = plot_conductivity_PT_surface(fig, [], [], [])
    assert artist is None
    assert ax.name != '3d', "Empty data should give a 2D message axes"
    assert len(ax.texts) > 0
    fig.canvas.draw()
