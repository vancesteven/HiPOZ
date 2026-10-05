#!/bin/bash
# Regenerate all Cortes 2026 figures with the corrected McCleskey model
# (total-ionic-strength fix + WATEQ4F speciation; audit 2026-10-05).
#
# Run on a machine with the full environment: TeX (figure fonts) and Reaktoro
# (speciation engages automatically via the 'auto' mode). The pre-fix figures
# are archived in cortes2026/figures_archive_20261005_pre_mccleskey_fix/ for
# side-by-side comparison.
#
# Usage:  ./regenerate_cortes_figures.sh

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "Cortes 2026 figure regeneration (McCleskey fix)"
echo "=============================================="

python3 - <<'EOF'
import speciation as spec
if spec.available():
    print("Reaktoro: available -- WATEQ4F speciation WILL be used for "
          "single salts with recipes (NaCl, KCl, MgSO4, Na2SO4, ...).")
else:
    print("WARNING: Reaktoro NOT available -- McCleskey curves fall back to "
          "total molality. MgSO4/Na2SO4 figures will NOT carry the "
          "speciation correction. Install with: "
          "conda install -c conda-forge reaktoro")
EOF

echo ""
echo "--- 1/2: benchtop study plots -> cortes2026/cortes_plots/ ---"
python3 cortes2026/cortes2026_plots.py

echo ""
echo "--- 2/2: Gamry EIS + McCleskey plots -> cortes2026/eis_plots/ ---"
python3 plot_cortes_with_mccleskey.py --show-mccleskey \
    --output-dir cortes2026/eis_plots

echo ""
echo "Done. Compare against cortes2026/figures_archive_20261005_pre_mccleskey_fix/"
