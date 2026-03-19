#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot configuration settings for conductivity studies.

This module contains universal plotting settings that can be applied to
all study plotting scripts (Mahboub et al., Cortes et al., etc.).
"""

# ========================================
# Font Sizes
# ========================================
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TITLE = 16
FONTSIZE_LEGEND = 10

# ========================================
# Colormaps
# ========================================
COLORMAP_CONCENTRATION = 'tab10'  # Discrete colormap for σ vs concentration plots
COLORMAP_TEMPERATURE = 'tab10'    # Discrete colormap for σ vs temperature plots
COLORMAP_PRESSURE = 'tab10'       # Discrete colormap for σ vs pressure plots

# ========================================
# Plot Options
# ========================================
SHOW_LEGEND = False  # Include legend in plots
SHOW_TITLE = True    # Include title in plots
SHOW_DELTA = True    # Include Delta (Δ%) deviation subplots

# ========================================
# Output Settings
# ========================================
DPI = 300  # Resolution for saved plots
