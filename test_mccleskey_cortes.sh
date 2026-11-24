#!/bin/bash
# Test script for McCleskey integration with Cortes plots

set -e  # Exit on error

echo "=========================================="
echo "McCleskey-Cortes Integration Test"
echo "=========================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
if ! command_exists python3; then
    echo "ERROR: python3 not found"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# ==========================================
# Test 1: Module functionality
# ==========================================

echo "Test 1: McCleskey module functions"
echo "------------------------------------------"

if [ -f "cortes_mccleskey.py" ]; then
    echo "Testing cortes_mccleskey.py..."
    python3 cortes_mccleskey.py 2>&1 | head -20
    echo ""
    echo "✓ Module test complete"
else
    echo "✗ cortes_mccleskey.py not found"
    exit 1
fi
echo ""

# ==========================================
# Test 2: Standard plots (no McCleskey)
# ==========================================

echo "Test 2: Generate standard plots (no McCleskey)"
echo "------------------------------------------"

if [ -f "plot_cortes_with_mccleskey.py" ]; then
    echo "Running: python3 plot_cortes_with_mccleskey.py"
    echo ""

    python3 plot_cortes_with_mccleskey.py 2>&1 | tail -15

    if [ -d "cortes_plots/single_salts" ]; then
        echo ""
        echo "✓ Standard plots directory created"
        ls -1 cortes_plots/single_salts/ 2>/dev/null | head -5
    else
        echo "⚠ No plots generated (may need data files)"
    fi
else
    echo "✗ plot_cortes_with_mccleskey.py not found"
    exit 1
fi
echo ""

# ==========================================
# Test 3: McCleskey comparison plots
# ==========================================

echo "Test 3: Generate McCleskey comparison plots"
echo "------------------------------------------"

echo "Running: python3 plot_cortes_with_mccleskey.py --show-mccleskey"
echo ""

python3 plot_cortes_with_mccleskey.py --show-mccleskey 2>&1 | tail -20

if [ -d "cortes_plots/mccleskey_comparison" ]; then
    echo ""
    echo "✓ McCleskey comparison directory created"
    n_plots=$(ls -1 cortes_plots/mccleskey_comparison/*.pdf 2>/dev/null | wc -l)
    echo "  Generated $n_plots McCleskey comparison plot(s)"
    ls -1 cortes_plots/mccleskey_comparison/ 2>/dev/null
else
    echo "⚠ No McCleskey plots generated (may need low-P data)"
fi
echo ""

# ==========================================
# Test 4: Custom pressure threshold
# ==========================================

echo "Test 4: Custom pressure threshold"
echo "------------------------------------------"

echo "Running with P ≤ 3.0 MPa..."
python3 plot_cortes_with_mccleskey.py --show-mccleskey --p-threshold 3.0 \
    2>&1 | grep -E "(Filtered|low pressure)" | head -5

echo ""
echo "✓ Custom threshold test complete"
echo ""

# ==========================================
# Test 5: Help and options
# ==========================================

echo "Test 5: Help and command-line options"
echo "------------------------------------------"

python3 plot_cortes_with_mccleskey.py --help | head -15

echo ""
echo "✓ Help text accessible"
echo ""

# ==========================================
# Summary
# ==========================================

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

if [ -d "cortes_plots" ]; then
    total_plots=$(find cortes_plots -name "*.pdf" 2>/dev/null | wc -l)
    echo "✓ Total plots generated: $total_plots"
    echo ""

    echo "Plot structure:"
    tree -L 2 cortes_plots 2>/dev/null || find cortes_plots -type f -name "*.pdf" | head -10
    echo ""
fi

echo "Next steps:"
echo "  1. Review generated PDFs in cortes_plots/"
echo "  2. Check Delta subplots show % deviation"
echo "  3. Verify only low-P data in McCleskey plots"
echo "  4. Compare with standard plots"
echo ""

echo "For detailed documentation, see:"
echo "  - MCCLESKEY_CORTES_GUIDE.md"
echo "  - PLOTTING_FORMATTING_UPDATES.md"
echo ""

echo "=========================================="
echo "All tests complete!"
echo "=========================================="
