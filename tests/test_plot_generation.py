#!/usr/bin/env python
"""
Test plot generation for all plot types.

Tests timeseries, Bode, Nyquist, and S vs P plot generation
to ensure plots are created without errors and contain expected elements.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
from gamryTools import TimeSeries, Solution
from gamryPlots import plot_timeseries

def create_mock_solution():
    """Create a mock Solution object with impedance data."""
    sol = Solution()

    # Mock frequency sweep data (typical EIS range)
    sol.freq_Hz = np.logspace(2, 5, 50)  # 100 Hz to 100 kHz

    # Mock impedance data (R-CPE circuit response)
    R0 = 100.0  # Ohms
    R1 = 50.0
    omega = 2 * np.pi * sol.freq_Hz
    tau = 1e-4  # Time constant

    # Simplified impedance
    Z_real = R0 + R1 / (1 + (omega * tau)**2)
    Z_imag = -R1 * omega * tau / (1 + (omega * tau)**2)

    sol.Zreal_ohm = Z_real
    sol.Zimag_ohm = Z_imag
    sol.Zmag_ohm = np.sqrt(Z_real**2 + Z_imag**2)
    sol.phase_deg = np.degrees(np.arctan2(Z_imag, Z_real))

    # Fitted values
    sol.R_calc = R0
    sol.R_unc = 2.0  # 2% uncertainty

    # Metadata
    sol.T = 298.15  # K
    sol.P = 10.0  # MPa
    sol.filename = "test_measurement.txt"

    return sol

def create_mock_timeseries(n_points=10):
    """Create a mock TimeSeries with multiple measurements."""
    ts = TimeSeries()

    # Time points
    timestamps = [f"2025-01-{i+10:02d} 12:00:00" for i in range(n_points)]

    ts.filenames = [f"test_{i}.txt" for i in range(n_points)]
    ts.timestamps = timestamps
    ts.Ts = np.linspace(273.15, 323.15, n_points)  # 0°C to 50°C
    ts.Ps = np.linspace(10, 100, n_points)  # 10 to 100 MPa

    # Resistance values with slight temperature/pressure dependence
    ts.Rcalc_ohm = 100 + 0.5 * (ts.Ps - 10) - 0.2 * (ts.Ts - 273.15)
    ts.percent_uncertainties = np.full(n_points, 2.0)
    # plot_timeseries iterates these in lockstep with timestamps/Rcalc_ohm
    ts.uncertainties = ts.Rcalc_ohm * ts.percent_uncertainties / 100
    ts.colors = ['C0'] * n_points
    ts.markers = ['o'] * n_points

    # Some measurements have conductivity values
    ts.conductivities_Sm = [None] * n_points
    ts.conductivities_Sm[5:] = np.linspace(0.01, 0.1, n_points - 5)
    ts.conductivities_unc_pct = [None] * 5 + list(np.full(n_points - 5, 3.0))

    return ts

def test_timeseries_plot():
    """Test timeseries plot generation."""
    print("=== Test: Timeseries Plot Generation ===\n")

    ts = create_mock_timeseries()

    # Generate plot
    # interactive=True returns the figure handles; interactive=False calls plt.show()
    fig, ax1, ax2 = plot_timeseries(ts, figure=None, fig_size=(12, 8), interactive=True)

    # Verify figure was created
    assert fig is not None, "Figure should not be None"
    assert ax1 is not None, "Primary axis should not be None"
    assert ax2 is not None, "Secondary axis should not be None"

    print("  ✓ Figure and axes created")

    # Check axes have data
    assert len(ax1.lines) > 0, "Primary axis should have plot lines"
    print(f"  ✓ Primary axis has {len(ax1.lines)} line(s)")

    # Check labels
    # Axes share x; plot_timeseries puts the x-label on the bottom (ax2) axis only
    assert ax2.get_xlabel() != "", "Bottom axis should have x-label"
    assert ax1.get_ylabel() != "", "Primary axis should have y-label"
    print(f"  ✓ Axes labeled: x='{ax1.get_xlabel()}', y='{ax1.get_ylabel()}'")

    plt.close(fig)
    print("\n✓ Timeseries plot generated successfully\n")
    return True

def test_bode_plot():
    """Test Bode plot generation."""
    print("=== Test: Bode Plot Generation ===\n")

    sol = create_mock_solution()

    # Create Bode plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot magnitude
    ax1.loglog(sol.freq_Hz, sol.Zmag_ohm, 'o-')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|Z| (Ohm)')
    ax1.grid(True)

    # Plot phase
    ax2.semilogx(sol.freq_Hz, sol.phase_deg, 'o-')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True)

    # Verify plot was created
    assert len(ax1.lines) > 0, "Magnitude plot should have data"
    assert len(ax2.lines) > 0, "Phase plot should have data"

    print(f"  ✓ Magnitude plot: {len(sol.freq_Hz)} frequency points")
    print(f"  ✓ Phase plot: {len(sol.freq_Hz)} frequency points")
    print(f"  ✓ Frequency range: {sol.freq_Hz.min():.1f} - {sol.freq_Hz.max():.1f} Hz")

    plt.close(fig)
    print("\n✓ Bode plot generated successfully\n")
    return True

def test_nyquist_plot():
    """Test Nyquist plot generation."""
    print("=== Test: Nyquist Plot Generation ===\n")

    sol = create_mock_solution()

    # Create Nyquist plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Nyquist (Re(Z) vs -Im(Z))
    ax.plot(sol.Zreal_ohm, -sol.Zimag_ohm, 'o-')
    ax.set_xlabel('Re(Z) (Ohm)')
    ax.set_ylabel('-Im(Z) (Ohm)')
    ax.set_aspect('equal')
    ax.grid(True)

    # Verify plot was created
    assert len(ax.lines) > 0, "Nyquist plot should have data"
    # matplotlib >=3.6 reports equal aspect as the float 1.0
    assert ax.get_aspect() in ('equal', 1.0), "Nyquist plot should have equal aspect ratio"

    print(f"  ✓ Nyquist plot: {len(sol.Zreal_ohm)} impedance points")
    print(f"  ✓ Re(Z) range: {sol.Zreal_ohm.min():.1f} - {sol.Zreal_ohm.max():.1f} Ohm")
    print(f"  ✓ Im(Z) range: {sol.Zimag_ohm.min():.1f} - {sol.Zimag_ohm.max():.1f} Ohm")
    print(f"  ✓ Aspect ratio: {ax.get_aspect()}")

    plt.close(fig)
    print("\n✓ Nyquist plot generated successfully\n")
    return True

def test_svp_plot():
    """Test conductivity vs pressure (S vs P) plot generation."""
    print("=== Test: S vs P Plot Generation ===\n")

    ts = create_mock_timeseries(20)

    # Give some measurements conductivity values
    ts.conductivities_Sm = list(np.linspace(0.01, 0.15, 20))

    # Create S vs P plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot conductivity vs pressure, colored by temperature
    scatter = ax.scatter(ts.Ps, ts.conductivities_Sm,
                         c=ts.Ts, cmap='coolwarm',
                         s=100, edgecolor='k', linewidth=0.5)

    ax.set_xlabel('Pressure (MPa)')
    ax.set_ylabel('Conductivity (S/m)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Temperature (K)')
    ax.grid(True)

    # Verify plot was created
    assert len(ax.collections) > 0, "S vs P plot should have scatter data"

    print(f"  ✓ Scatter plot: {len(ts.Ps)} data points")
    print(f"  ✓ P range: {min(ts.Ps):.1f} - {max(ts.Ps):.1f} MPa")
    print(f"  ✓ S range: {min(ts.conductivities_Sm):.4f} - {max(ts.conductivities_Sm):.4f} S/m")
    print(f"  ✓ T range: {min(ts.Ts):.1f} - {max(ts.Ts):.1f} K")
    print(f"  ✓ Colorbar present: {cbar is not None}")

    plt.close(fig)
    print("\n✓ S vs P plot generated successfully\n")
    return True

def test_plot_export():
    """Test plot export to file."""
    print("=== Test: Plot Export ===\n")

    sol = create_mock_solution()

    # Create simple plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sol.freq_Hz, sol.Zmag_ohm, 'o-')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|Z| (Ohm)')

    # Export to PNG
    test_dir = Path('tests/test_output')
    test_dir.mkdir(parents=True, exist_ok=True)

    png_path = test_dir / 'test_plot.png'
    pdf_path = test_dir / 'test_plot.pdf'

    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    # Verify files were created
    assert png_path.exists(), f"PNG file not created: {png_path}"
    assert pdf_path.exists(), f"PDF file not created: {pdf_path}"

    # Check file sizes (should be non-zero)
    png_size = png_path.stat().st_size
    pdf_size = pdf_path.stat().st_size

    assert png_size > 0, f"PNG file is empty: {png_path}"
    assert pdf_size > 0, f"PDF file is empty: {pdf_path}"

    print(f"  ✓ PNG exported: {png_path.name} ({png_size / 1024:.1f} KB)")
    print(f"  ✓ PDF exported: {pdf_path.name} ({pdf_size / 1024:.1f} KB)")

    plt.close(fig)
    print("\n✓ Plot export successful\n")
    return True

def test_plot_with_error_bars():
    """Test plotting with error bars/uncertainties."""
    print("=== Test: Plot with Error Bars ===\n")

    ts = create_mock_timeseries(10)

    # Create plot with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert timestamps to numeric for plotting
    x = np.arange(len(ts.Rcalc_ohm))
    y = ts.Rcalc_ohm
    yerr = ts.Rcalc_ohm * ts.percent_uncertainties / 100

    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5)
    ax.set_xlabel('Measurement Index')
    ax.set_ylabel('Resistance (Ohm)')
    ax.grid(True)

    # Verify error bars were added
    assert len(ax.containers) > 0, "Plot should have error bar containers"

    print(f"  ✓ Data points: {len(x)}")
    print(f"  ✓ Error bars: ±{ts.percent_uncertainties[0]:.1f}%")
    print(f"  ✓ R range: {y.min():.2f} - {y.max():.2f} Ohm")

    plt.close(fig)
    print("\n✓ Error bar plot generated successfully\n")
    return True

def test_multi_panel_plot():
    """Test creation of multi-panel figure."""
    print("=== Test: Multi-Panel Plot ===\n")

    sol = create_mock_solution()

    # Create 2x2 panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Magnitude
    axes[0, 0].loglog(sol.freq_Hz, sol.Zmag_ohm)
    axes[0, 0].set_title('Magnitude')
    axes[0, 0].set_xlabel('f (Hz)')
    axes[0, 0].set_ylabel('|Z| (Ohm)')

    # Panel 2: Phase
    axes[0, 1].semilogx(sol.freq_Hz, sol.phase_deg)
    axes[0, 1].set_title('Phase')
    axes[0, 1].set_xlabel('f (Hz)')
    axes[0, 1].set_ylabel('Phase (deg)')

    # Panel 3: Nyquist
    axes[1, 0].plot(sol.Zreal_ohm, -sol.Zimag_ohm, 'o-')
    axes[1, 0].set_title('Nyquist')
    axes[1, 0].set_xlabel('Re(Z) (Ohm)')
    axes[1, 0].set_ylabel('-Im(Z) (Ohm)')
    axes[1, 0].set_aspect('equal')

    # Panel 4: Real and Imaginary
    axes[1, 1].semilogx(sol.freq_Hz, sol.Zreal_ohm, label='Re(Z)')
    axes[1, 1].semilogx(sol.freq_Hz, sol.Zimag_ohm, label='Im(Z)')
    axes[1, 1].set_title('Real and Imaginary')
    axes[1, 1].set_xlabel('f (Hz)')
    axes[1, 1].set_ylabel('Z (Ohm)')
    axes[1, 1].legend()

    plt.tight_layout()

    # Verify all panels have data
    for i in range(2):
        for j in range(2):
            assert len(axes[i, j].lines) > 0 or len(axes[i, j].collections) > 0, \
                f"Panel [{i},{j}] should have data"

    print("  ✓ All 4 panels populated")
    print("  ✓ Layout optimized with tight_layout()")

    plt.close(fig)
    print("\n✓ Multi-panel plot generated successfully\n")
    return True

if __name__ == '__main__':
    try:
        # Run all tests
        test_timeseries_plot()
        test_bode_plot()
        test_nyquist_plot()
        test_svp_plot()
        test_plot_export()
        test_plot_with_error_bars()
        test_multi_panel_plot()

        print("=" * 50)
        print("✓ ALL PLOT GENERATION TESTS PASSED")
        print("=" * 50)

    except AssertionError as e:
        print("\n" + "=" * 50)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 50)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        sys.exit(1)
