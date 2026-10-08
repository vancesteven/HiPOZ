#!/bin/bash
# Unified plotting script for HiPOZ conductivity studies
# Runs both Gamry-only and benchtop+Gamry plotting workflows

set -e  # Exit on error

echo "=========================================="
echo "HiPOZ Conductivity Plotting Suite"
echo "=========================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
if ! command_exists python3; then
    echo "ERROR: python3 not found. Please install Python 3."
    exit 1
fi

# Parse arguments
STUDY="${1:-all}"
FORMAT="${2:-pdf}"

# ==========================================
# 1. Cortes Gamry-only plots (Current implementation)
# ==========================================

if [ "$STUDY" = "cortes" ] || [ "$STUDY" = "all" ]; then
    echo "=========================================="
    echo "Cortes et al. (2026) - Gamry Impedance Plots"
    echo "=========================================="
    echo ""
    echo "Generating 5 publication plots..."
    echo ""

    python3 plot_cortes_publication.py 20250813Cortes 20250814Cortes 20250815Cortes \
        --output-dir cortes_plots

    echo ""
    echo "✓ Cortes Gamry plots complete!"
    echo "  Output: cortes_plots/"
    echo ""
fi

# ==========================================
# 2. Mahboub benchtop+Gamry plots (Full workflow)
# ==========================================

if [ "$STUDY" = "mahboub" ] || [ "$STUDY" = "all" ]; then
    echo "=========================================="
    echo "Mahboub et al. (2026) - Benchtop + Gamry Overlay"
    echo "=========================================="
    echo ""

    if [ -f "mahboub2026/mahboub2026_plots.py" ]; then
        echo "Generating compound-specific plots with McCleskey overlay..."
        echo ""

        cd mahboub2026
        python3 mahboub2026_plots.py
        cd ..

        echo ""
        echo "✓ Mahboub plots complete!"
        echo "  Output: mahboub2026/mahboub_plots/"
        echo ""
    else
        echo "⚠️  Mahboub plotting script not found, skipping..."
        echo ""
    fi
fi

# ==========================================
# 3. Summary
# ==========================================

echo "=========================================="
echo "Plotting Complete!"
echo "=========================================="
echo ""

if [ "$STUDY" = "cortes" ] || [ "$STUDY" = "all" ]; then
    echo "Cortes plots:"
    echo "  - cortes_plots/single_salts/ (3 plots)"
    echo "  - cortes_plots/mixtures/ (2 plots)"
    echo ""
fi

if [ "$STUDY" = "mahboub" ] || [ "$STUDY" = "all" ]; then
    if [ -d "mahboub2026/mahboub_plots" ]; then
        echo "Mahboub plots:"
        echo "  - mahboub2026/mahboub_plots/ (10 plots)"
        echo ""
    fi
fi

echo "Usage:"
echo "  ./plot_all.sh           # Plot both studies"
echo "  ./plot_all.sh cortes    # Cortes only"
echo "  ./plot_all.sh mahboub   # Mahboub only"
echo ""
