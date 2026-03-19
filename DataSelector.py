import sys
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QListWidget, QLabel, QMessageBox, QComboBox, QFileDialog,
                             QTableWidget, QTableWidgetItem, QTabWidget, QHBoxLayout, QStatusBar)
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from gamryPlots import plot_timeseries
import logging

# Get logger
log = logging.getLogger('HiPOZ')

class DataSelector(QMainWindow):
    def __init__(self, timeseries, calib_config=None):
        super(DataSelector,self).__init__()
        self.timeseries = timeseries  # This is the array of Solution objects supplied externally
        self.calib_config = calib_config  # Optional calibration configuration
        self.updating_table = False
        self.selected_points = []
        self.ax1 = []
        self.ax2 = []
        self.config_file_paths = {}  # Track config file paths for each date
        self.init_ui()
        self.current_std = []
        self.current_std_uncertainty = []
        self.current_meas = []

        # Check for config files in correct locations (following zAnalysis<date>.json convention)
        # Auto-creates missing files to store GUI progress
        self.check_config_file_locations()

        # Auto-apply calibration config if provided
        if self.calib_config:
            self.apply_calibration_config()
    def init_ui(self):
        self.setGeometry(200, 200, 1000, 800)
        self.setWindowTitle('Gamry Data')

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create the tab widget
        self.tabs = QTabWidget()
        self.timeseries_tab = QWidget()
        self.plots_tab = QWidget()

        # Layout for the S vs P tab
        self.svp_tab = QWidget()
        self.svp_layout = QVBoxLayout()

        # Create figure and canvas for S vs P
        self.svp_figure = Figure()
        self.svp_canvas = FigureCanvas(self.svp_figure)
        self.svp_layout.addWidget(self.svp_canvas)
        self.svp_tab.setLayout(self.svp_layout)

        # Layout for the timeseries tab
        self.timeseries_layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.timeseries_layout.addWidget(self.canvas)
        self.timeseries_tab.setLayout(self.timeseries_layout)

        # Layout for the plots tab
        self.plots_layout = QVBoxLayout()

        # Create figures and canvases for Bode and Nyquist plots
        self.bode_figure = Figure()
        self.bode_canvas = FigureCanvas(self.bode_figure)
        self.nyquist_figure = Figure()
        self.nyquist_canvas = FigureCanvas(self.nyquist_figure)

        # Add canvases to the layout
        self.plots_layout.addWidget(self.bode_canvas)
        self.plots_layout.addWidget(self.nyquist_canvas)
        self.plots_tab.setLayout(self.plots_layout)

        # Add tabs to the widget
        self.tabs.addTab(self.timeseries_tab, "Timeseries")
        self.tabs.addTab(self.plots_tab, "Bode & Nyquist")
        self.tabs.addTab(self.svp_tab, "S vs P")

        # Layout for table and buttons
        self.table_layout = QVBoxLayout()

        # Main layout
        main_layout = QVBoxLayout()

        # Layout for table and buttons
        self.table_layout = QVBoxLayout()

        # Extract just filenames (not full paths) for display
        filenames_display = [Path(f).name for f in self.timeseries.filenames]

        # Replace zero values with None for P and T (zero doesn't make physical sense)
        Ts_display = [T if T != 0 else None for T in self.timeseries.Ts]
        Ps_display = [P if P != 0 else None for P in self.timeseries.Ps]

        self.data = pd.DataFrame({
            'Filename': filenames_display,
            'Calibration': [None] * len(filenames_display),  # Shows calibration group and role
            'Time': self.timeseries.timestamps,
            'Comp': [None] * len(filenames_display),  # Leave blank by default
            'w (ppt)': [None] * len(filenames_display),  # Leave blank by default
            'T (K)': Ts_display,
            'P (MPa)': Ps_display,
            'Z (Ohm)': self.timeseries.Rcalc_ohm,
            'Z± (Ohm)': self.timeseries.percent_uncertainties,
            'S (S/m)': self.timeseries.conductivities_Sm,
            'S± (S/m)': self.timeseries.conductivities_unc_pct
        })
        self.associated_mask = np.zeros(len(self.data), dtype=bool)  # rows marked by Associate Measurements
        # Cast 'P (MPa)' to int where possible, keep as float otherwise to handle NaN
        # This allows display of both integer pressures and NaN values
        try:
            # Try to convert, but if there are NaN values, keep as float
            p_vals = pd.to_numeric(self.data['P (MPa)'], errors='coerce')
            # Only convert to int if no NaN values
            if not p_vals.isna().any():
                self.data['P (MPa)'] = p_vals.astype(int)
            else:
                self.data['P (MPa)'] = p_vals  # Keep as float to preserve NaN
        except Exception:
            pass  # Keep original values if conversion fails
        # Pandas DataFrame Display as a Table
        self.table = QTableWidget()
        # Setting up the table for row selection
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.MultiSelection)  # SingleSelection or MultiSelection
        self.table.setColumnCount(len(self.data.columns))
        self.table.setRowCount(len(self.data.index))
        self.table.setHorizontalHeaderLabels(self.data.columns)
        self.refresh_table()
        self.table.itemChanged.connect(self.on_table_item_changed)

        # Add buttons
        self.btn_clear_selection = QPushButton('Clear Selections')
        self.btn_mark_standard = QPushButton('Mark as Standard')
        self.btn_associate_measurements = QPushButton('Associate Measurements')
        self.btn_create_plots = QPushButton('Create Bode and Nyquist Plots')
        self.btn_export_plots = QPushButton('Export Plots to PDF')

        # Connect buttons to functions
        self.btn_clear_selection.clicked.connect(self.clear_table_selection)
        self.btn_mark_standard.clicked.connect(self.mark_as_standard)
        self.btn_associate_measurements.clicked.connect(self.associate_measurements)
        self.btn_create_plots.clicked.connect(self.create_plots)
        self.btn_export_plots.clicked.connect(self.export_plots)

        # Add widgets to table layout
        self.table_layout.addWidget(self.table)
        self.table_layout.addWidget(self.btn_clear_selection)
        self.table_layout.addWidget(self.btn_mark_standard)
        self.table_layout.addWidget(self.btn_associate_measurements)
        self.table_layout.addWidget(self.btn_create_plots)
        self.table_layout.addWidget(self.btn_export_plots)

        # Create a horizontal layout to combine table layout and tab widget
        combined_layout = QHBoxLayout()
        combined_layout.addLayout(self.table_layout)
        combined_layout.addWidget(self.tabs)

        main_layout.addLayout(combined_layout)

        # Create a central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Plot data
        self.plot_timeseries()

        # Set export directory to config file location if available, otherwise hipoz_exports
        if self.calib_config and hasattr(self.calib_config, 'config_path') and self.calib_config.config_path:
            self.export_dir = Path(self.calib_config.config_path).parent
            log.info(f"Using config directory for exports: {self.export_dir}")
        else:
            self.export_dir = Path.cwd() / "hipoz_exports"
            log.info("No config path found, using hipoz_exports/")

        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.save_mode = "overwrite"  # "overwrite" | "timestamp" | "rolling"
        self.rolling_keep = 5  # used only for "rolling"

        # self.annotation = self.ax1.annotate('Highlighted',
        #                                     xy=(0, 0), xytext=(20, 20),
        #                                     textcoords='offset points',
        #                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        #                                     bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5))
        # self.annotation.set_visible(False)
        # self.tooltip = self.ax1.text(0, 0, "", va="bottom", ha="left")
        # self.rect = plt.Rectangle((0, 0), 1, 1, edgecolor='yellow', facecolor='none')
        # self.ax1.add_patch(self.rect)
        # self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        # self.canvas.mpl_connect('pick_event', self.on_pick)  # Connect the pick event

    def plot_timeseries(self):
        # Ensure this function exists and is correctly referenced
        try:
            self.figure, self.ax1, self.ax2 = plot_timeseries(self.timeseries, figure=self.figure, fig_size=(26, 14), interactive=True)
            # QMessageBox.information(self, "Plotting", "Plot generated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.draw()  # Redraw the canvas
        self.show()

    def on_hover(self, event):
        if event.inaxes == self.ax1:
            min_dist = float('inf')
            index = None
            x = mdates.date2num(self.timeseries.timestamps)
            y = self.timeseries.Rcalc_ohm
            for i, (xi, yi) in enumerate(zip(x,y)):
                dist = np.sqrt((xi - event.xdata) ** 2 + (yi - event.ydata) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
                    # print(min_dist) # debugging
            if closest_index is not None and min_dist < 20:  # Sensitivity threshold
                x, y = x[closest_index], y[closest_index]
                point_type = "Calibration" if self.determine_point_type(closest_index) else "Measurement"
                self.annotation.xy = (x, y)
                self.annotation.set_text(f'{point_type}: {int(self.timeseries.ws_ppt[closest_index])} ppt')
                self.annotation.get_bbox_patch().set_alpha(0.4)
                self.annotation.set_visible(True)
                self.figure.canvas.draw_idle()
            else:
                self.annotation.set_visible(False)
            self.figure.canvas.draw_idle()

    def determine_point_type(self, ind):
        if 'Std' in self.timeseries.filenames[ind]:
            return True  # Example condition

    def on_table_item_changed(self, item):
        if self.updating_table:
            return
        row = item.row()
        col = item.column()
        header = self.data.columns[col]
        text = item.text()

        # Parse types like your DataFrame expects
        try:
            if header in ['w (ppt)', 'T (K)', 'Z (Ohm)', 'Z± (Ohm)', 'S (S/m)', 'S± (S/m)']:
                # Allow empty/None values for editable fields
                val = float(text) if text and text.strip() and text.lower() != 'none' else None
            elif header in ['P (MPa)']:
                val = int(float(text)) if text and text.strip() and text.lower() != 'none' else None
            elif header in ['Comp']:
                # String field - allow empty
                val = text if text and text.strip() and text.lower() != 'none' else None
            else:
                val = text
        except ValueError:
            # Revert bad input
            self.updating_table = True
            item.setText(str(self.data.iloc[row, col]))
            self.updating_table = False
            return

        # Update DataFrame
        self.data.iat[row, col] = val

        # (Optional) keep the underlying timeseries arrays in sync for key columns
        # Skip 'Filename' column - it's read-only
        try:
            if header == 'Filename':
                # Revert any edits to filename column
                self.updating_table = True
                item.setText(str(self.data.iloc[row, col]))
                self.updating_table = False
                return
            elif header == 'Z (Ohm)':
                self.timeseries.Rcalc_ohm[row] = float(val) if val is not None else np.nan
            elif header == 'S (S/m)':
                self.timeseries.conductivities_Sm[row] = float(val) if val is not None else None
            elif header == 'P (MPa)':
                self.timeseries.Ps[row] = int(val) if val is not None else np.nan
            elif header == 'T (K)':
                self.timeseries.Ts[row] = float(val) if val is not None else np.nan
            elif header == 'w (ppt)':
                self.timeseries.ws_ppt[row] = float(val) if val is not None else None
            elif header == 'Comp':
                self.timeseries.comp[row] = val
        except Exception:
            pass

    def update_annotation(self, x, y):
        self.annotation.xy = (x, y)
        self.annotation.set_text(f'({x}, {y})')
        self.annotation.set_visible(True)
        self.figure.canvas.draw_idle()

    def on_pick(self, event):
        artist = event.artist
        xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        xdata, ydata = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        print('Selected data:', np.take(self.timeseries.filenames, ind))
        point = np.take(self.timeseries, ind)[0]  # Take first in case of multiple points
        self.selected_points.append(point)

    # def mark_as_standard(self):
    #     selected_indexes = self.table.selectionModel().selectedRows()
    #     cell_consts = []
    #     for index in selected_indexes:
    #         row = index.row()
    #         this_cond = self.timeseries.conductivities_Sm[row]
    #         this_imp = self.timeseries.Rcalc_ohm[row]
    #         this_cell_const = this_cond/this_imp
    #         print(f"Marking row {row} as standard. S: {this_cond}; Z: {this_imp}; cell constant: {this_cell_const}")
    #         cell_consts.append(this_cell_const)
    #     print(f"cell constants are as follows: {cell_consts}")
    #     self.current_std = np.mean(cell_consts)
    #     print(f"using the mean as {self.current_std} in units of 1/m")
    #     self.clear_table_selection()  # Clear selection after processing

    def mark_as_standard(self):
        selected_indexes = self.table.selectionModel().selectedRows()
        if not selected_indexes:
            QMessageBox.warning(self, "No Selection", "Select at least one row.")
            return

        col_S = self.data.columns.get_loc('S (S/m)')
        col_Z = self.data.columns.get_loc('Z (Ohm)')
        col_dZ = self.data.columns.get_loc('Z± (Ohm)')

        cell_consts = []
        cell_const_uncertainties = []
        for index in selected_indexes:
            row = index.row()

            # Read from the table to use the latest edited values
            S_item = self.table.item(row, col_S)
            Z_item = self.table.item(row, col_Z)
            dZ_item = self.table.item(row, col_dZ)
            try:
                this_cond = float(S_item.text()) if S_item is not None else float(self.data.iat[row, col_S])
                this_imp = float(Z_item.text()) if Z_item is not None else float(self.data.iat[row, col_Z])
                this_dZ = float(dZ_item.text()) if dZ_item is not None else float(self.data.iat[row, col_dZ])
            except (TypeError, ValueError):
                QMessageBox.warning(self, "Bad Value", f"Row {row}: cannot parse S or Z.")
                continue

            if this_imp == 0:
                QMessageBox.warning(self, "Bad Value", f"Row {row}: Z (Ohm) is zero.")
                continue

            this_cell_const = this_cond * this_imp
            this_unc_cell_const = this_dZ
            print(f"Marking row {row} as standard. S: {this_cond}; Z: {this_imp}; cell constant: {this_cell_const}")
            cell_consts.append(this_cell_const)
            cell_const_uncertainties.append(this_unc_cell_const)

        if not cell_consts:
            QMessageBox.warning(self, "No Valid Rows", "Provide valid S and Z values.")
            return
        n = np.size(cell_consts)
        s = np.std(cell_consts,ddof=1)
        self.current_std = float(np.mean(cell_consts))
        if n>1:
            self.current_std_unc = np.sqrt(np.mean(np.power(cell_const_uncertainties,2))/len(cell_const_uncertainties) + s**2/n)
        else:
            self.current_std_unc = cell_const_uncertainties
        print(f"using the mean as {self.current_std} in units of 1/m")
        self.clear_table_selection()

        # Save GUI state to config file
        self.save_gui_state_to_config()

    # def associate_measurements(self):
    #     selected_indexes = self.table.selectionModel().selectedRows()
    #     for index in selected_indexes:
    #         row = index.row()
    #         this_cond = self.current_std/self.timeseries.Rcalc_ohm[row]
    #         print(f"Associating measurement for row {row}. S: {this_cond}")
    #         self.data.at[row, 'S (S/m)'] = this_cond  # Update DataFrame
    #         self.timeseries.conductivities_Sm[row] = this_cond
    #         self.associated_mask[row] = True
    #     self.refresh_table()
    #     self.refresh_s_vs_p_plot()
    #     # Save exactly once here
    #     self.save_curated_outputs()

    def associate_measurements(self):
        """
        Recompute association strictly from the *current* selection:
        - Clear previous associations
        - For selected rows: recompute S using current_std and (latest) Z
        - Mark only selected rows as associated=True
        - Refresh table, S–P plot, and save once
        """
        sel = self.table.selectionModel().selectedRows()

        # If nothing selected, clear all associations
        n = len(self.data)
        if not hasattr(self, "associated_mask") or len(self.associated_mask) != n:
            self.associated_mask = np.zeros(n, dtype=bool)
        else:
            self.associated_mask[:] = False  # clear previous associations

        if not sel:
            # nothing associated → empty S–P (since plot uses associated_mask)
            self.refresh_table()
            self.refresh_s_vs_p_plot()
            self.save_curated_outputs()
            return

        # columns
        col_Z = self.data.columns.get_loc('Z (Ohm)')
        col_dZ = self.data.columns.get_loc('Z± (Ohm)')
        col_S = self.data.columns.get_loc('S (S/m)')
        col_dS = self.data.columns.get_loc('S± (S/m)')

        for idx in sel:
            row = idx.row()

            # Prefer latest table value for Z
            Z_item = self.table.item(row, col_Z)
            dz_item = self.table.item(row, col_dZ)
            try:
                R_val = float(Z_item.text()) if Z_item is not None else float(self.data.iat[row, col_Z])
                dz_val = float(dz_item.text()) if dz_item is not None else float(self.data.iat[row, col_dZ])
            except Exception:
                # fall back to timeseries value
                R_val = float(self.timeseries.Rcalc_ohm[row])
                dz_val = float(self.timeseries.uncertainties[row])

            if R_val == 0:
                continue
            this_std = float(self.current_std)
            S_val = this_std / R_val
            dZ = dz_val
            dS_val = np.sqrt((S_val*dZ / R_val) ** 2 + (this_std*dZ/R_val**2)**2)

            # Update DataFrame + timeseries
            self.data.iat[row, col_S] = S_val
            self.data.iat[row, col_dS] = dS_val
            try:
                self.timeseries.conductivities_Sm[row] = S_val
            except Exception:
                pass

            # Mark associated
            self.associated_mask[row] = True

        # Update UI + plot + save (once)
        self.refresh_table()
        self.refresh_s_vs_p_plot()
        self.save_curated_outputs()

        # Save GUI state to config file
        self.save_gui_state_to_config()

    def clear_table_selection(self):
        self.table.selectionModel().clearSelection()

    def refresh_table(self):
        self.updating_table = True
        try:
            self.table.setRowCount(len(self.data))
            self.table.setColumnCount(len(self.data.columns))
            self.table.setHorizontalHeaderLabels(self.data.columns)

            for i, row in self.data.iterrows():
                for j, col in enumerate(self.data.columns):
                    val = row[col]
                    if pd.isna(val) or val is None:
                        formatted_value = ''  # Show empty for None/NaN
                    elif isinstance(val, pd.Timestamp):
                        formatted_value = val.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        formatted_value = str(val)
                    self.table.setItem(i, j, QTableWidgetItem(formatted_value))

            self.table.resizeColumnsToContents()
        finally:
            self.updating_table = False

    # def create_plots(self):
    #     selected_indexes = self.table.selectionModel().selectedRows()
    #     if not selected_indexes:
    #         QMessageBox.warning(self, "No Selection", "No data points selected for plotting.")
    #         return
    #
    #     # Extract the selected data
    #     frequencies = []
    #     impedances = []
    #     for index in selected_indexes:
    #         row = index.row()
    #         frequencies.append(self.timeseries.frequencies[row])
    #         impedances.append(self.timeseries.impedances[row])
    #
    #     frequencies = np.array(frequencies)
    #     impedances = np.array(impedances)
    #
    #     # Bode Plot
    #     self.bode_figure.clear()
    #     ax_bode_magnitude = self.bode_figure.add_subplot(211)
    #     # ax_bode_magnitude.set_title("Bode Plot - Magnitude")
    #     ax_bode_magnitude.set_xlabel("Frequency (Hz)")
    #     ax_bode_magnitude.set_ylabel("Impedance Magnitude (Ohms)")
    #     ax_bode_magnitude.set_xscale('log')  # Set log scale for x-axis
    #
    #     ax_bode_phase = self.bode_figure.add_subplot(212)
    #     # ax_bode_phase.set_title("Bode Plot - Phase")
    #     ax_bode_phase.set_xlabel("Frequency (Hz)")
    #     ax_bode_phase.set_ylabel("Phase (degrees)")
    #     ax_bode_phase.set_xscale('log')  # Set log scale for x-axis
    #
    #     # Nyquist Plot
    #     self.nyquist_figure.clear()
    #     ax_nyquist = self.nyquist_figure.add_subplot(111)
    #     ax_nyquist.set_title("Nyquist Plot")
    #     ax_nyquist.set_xlabel("Real Part (Ohms)")
    #     ax_nyquist.set_ylabel("Imaginary Part (Ohms)")
    #
    #     # Plot each selected dataset separately
    #     for index in selected_indexes:
    #         row = index.row()
    #         frequencies = self.timeseries.frequencies[row]
    #         impedances = self.timeseries.impedances[row]
    #         fits = self.timeseries.impedance_fits[row]
    #
    #         # Bode Plot - Magnitude and Phase
    #         ax_bode_magnitude.plot(frequencies, np.abs(impedances), marker='o', linestyle='', label=f'Data {row}')
    #         ax_bode_magnitude.plot(frequencies, np.abs(fits), marker='', linestyle='-')
    #         ax_bode_phase.plot(frequencies, np.angle(impedances, deg=True), marker='o', linestyle='',
    #                            label=f'Data {row}')
    #         ax_bode_phase.plot(frequencies, np.angle(fits, deg=True), marker='', linestyle='-')
    #
    #         # Nyquist Plot
    #         ax_nyquist.plot(np.real(impedances), -np.imag(impedances), marker='o', linestyle='', label=f'Data {row}')
    #         ax_nyquist.plot(np.real(fits), -np.imag(fits), marker='', linestyle='-', label=f'Data {row}')
    #
    #     # Position legends outside the plots on the right
    #     ax_bode_magnitude.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     # ax_bode_phase.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     ax_nyquist.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #
    #     self.bode_canvas.draw()
    #     self.nyquist_canvas.draw()
    #     self.tabs.setCurrentIndex(1)
    # # def refresh_table(self):
    # #     self.table.setRowCount(len(self.data))
    # #     for row in range(len(self.data)):
    # #         for col in range(len(self.data.columns)):
    # #             item = QTableWidgetItem(str(self.data.iloc[row, col]))
    # #             self.table.setItem(row, col, item)
    # #     self.table.resizeColumnsToContents()  # Resize columns to fit content

    def create_plots(self):
        import traceback

        try:
            selected_indexes = self.table.selectionModel().selectedRows()
            if not selected_indexes:
                QMessageBox.warning(self, "No Selection", "No data points selected for plotting.")
                return

            # Clear figures / axes
            self.bode_figure.clear()
            ax_bode_mag = self.bode_figure.add_subplot(211)
            ax_bode_phase = self.bode_figure.add_subplot(212)

            self.nyquist_figure.clear()
            ax_nyq = self.nyquist_figure.add_subplot(111)

            # Labels/scales
            ax_bode_mag.set_xlabel("Frequency (Hz)")
            ax_bode_mag.set_ylabel("Impedance Magnitude ($\Omega$)")
            ax_bode_mag.set_xscale('log')

            ax_bode_phase.set_xlabel("Frequency (Hz)")
            ax_bode_phase.set_ylabel("Phase (deg)")
            ax_bode_phase.set_xscale('log')

            ax_nyq.set_title("Nyquist Plot")
            ax_nyq.set_xlabel("Real(Z) ($\Omega$)")
            ax_nyq.set_ylabel("−Imag(Z) ($\Omega$)")

            # One color per dataset, reused for data + fit
            color_list = plt.rcParams['axes.prop_cycle'].by_key().get(
                'color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
            )
            color_iter = iter(color_list)

            for index in selected_indexes:
                row = index.row()
                P = np.asarray(self.timeseries.Ps[row])
                T = np.asarray(self.timeseries.Ts[row])

                freqs = np.asarray(self.timeseries.frequencies[row])
                Z = np.asarray(self.timeseries.impedances[row])
                Zfit = np.asarray(self.timeseries.impedance_fits[row])

                if freqs.size == 0 or Z.size == 0:
                    continue

                try:
                    color = next(color_iter)
                except StopIteration:
                    color_iter = iter(color_list)
                    color = next(color_iter)

                label = f"P:{P} MPa, T:{T} K"

                # Bode magnitude
                ax_bode_mag.plot(freqs, np.abs(Z), marker='o', linestyle='', color=color, label=label)
                if Zfit.size == Z.size:
                    ax_bode_mag.plot(freqs, np.abs(Zfit), linestyle='-', color=color)

                # Bode phase
                ax_bode_phase.plot(freqs, np.angle(Z, deg=True), marker='o', linestyle='', color=color, label=label)
                if Zfit.size == Z.size:
                    ax_bode_phase.plot(freqs, np.angle(Zfit, deg=True), linestyle='-', color=color)

                # Nyquist
                ax_nyq.plot(np.real(Z), -np.imag(Z), marker='o', linestyle='', color=color, label=label)
                if Zfit.size == Z.size:
                    ax_nyq.plot(np.real(Zfit), -np.imag(Zfit), linestyle='-', color=color)

            # Legends + grid
            ax_bode_mag.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax_nyq.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            for ax in (ax_bode_mag, ax_bode_phase, ax_nyq):
                ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

            self.bode_canvas.draw()
            self.nyquist_canvas.draw()
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            tb = traceback.format_exc()
            try:
                log.error("Error in create_plots:\n%s", tb)
            except Exception:
                # if 'log' isn't configured, at least print
                print("Error in create_plots:\n", tb)

            # Nice Qt dialog with expandable details
            m = QMessageBox(self)
            m.setIcon(QMessageBox.Critical)
            m.setWindowTitle("Plotting error")
            m.setText(str(e))
            m.setInformativeText("An error occurred while generating plots.")
            m.setDetailedText(tb)
            m.exec_()

    def refresh_s_vs_p_plot(self):
        """
        Scatter S vs P colored by temperature (°C), fixed color scale from -20 to 80 °C.
        Uses self.data so it reflects table edits and associated measurements.
        """
        # Guard if data is missing
        if self.data is None or 'S (S/m)' not in self.data or 'P (MPa)' not in self.data or 'T (K)' not in self.data:
            return

        # Pull arrays from the DataFrame (reflects table edits)
        try:
            S = pd.to_numeric(self.data['S (S/m)'], errors='coerce').to_numpy()
            P = pd.to_numeric(self.data['P (MPa)'], errors='coerce').to_numpy()
            T_K = pd.to_numeric(self.data['T (K)'], errors='coerce').to_numpy()
        except Exception:
            return

        # Convert to °C
        T_C = T_K - 273.15

        # Keep only finite rows
        mask = np.isfinite(S) & np.isfinite(P) & np.isfinite(T_C) & self.associated_mask
        S = S[mask]
        P = P[mask]
        T_C = T_C[mask]

        # Clear and redraw
        self.svp_figure.clear()
        ax = self.svp_figure.add_subplot(111)

        # Fixed color scale -20 to 80 °C
        # vmin, vmax = -20.0, 80.0
        vmin = np.nanmin(T_C)
        vmax = np.nanmax(T_C)
        sc = ax.scatter(P, S, c=T_C, cmap='viridis', vmin=vmin, vmax=vmax, edgecolors='none')

        ax.set_xlabel("P (MPa)")
        ax.set_ylabel("S (S/m)")
        ax.set_title("Conductivity vs Pressure")

        # Colorbar
        cbar = self.svp_figure.colorbar(sc, ax=ax)
        cbar.set_label("Temperature (°C)")

        # Nice grid
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.7)

        self.svp_canvas.draw()
        # auto-save curated data and plots
        # self.save_curated_outputs()

    def _ensure_mask_shape(self):
        n = len(self.data)
        if not hasattr(self, "associated_mask"):
            self.associated_mask = np.zeros(n, dtype=bool)
            return
        if len(self.associated_mask) < n:
            # extend with False
            extra = np.zeros(n - len(self.associated_mask), dtype=bool)
            self.associated_mask = np.concatenate([self.associated_mask, extra])
        elif len(self.associated_mask) > n:
            # truncate
            self.associated_mask = self.associated_mask[:n]

    def export_plots(self):
        base, _ = QFileDialog.getSaveFileName(
            self, "Save base filename (no extension)",
            str(self.export_dir / "hipoz_export"),
            "All Files (*)"
        )
        if not base:
            return
        basepath = Path(base)
        # save what's currently drawn; no recompute
        try:
            if hasattr(self, "bode_figure") and len(self.bode_figure.axes) > 0:
                self.bode_figure.savefig(str(basepath) + "_Bode.pdf", bbox_inches="tight")
            if hasattr(self, "nyquist_figure") and len(self.nyquist_figure.axes) > 0:
                self.nyquist_figure.savefig(str(basepath) + "_Nyquist.pdf", bbox_inches="tight")
            if hasattr(self, "svp_figure") and len(self.svp_figure.axes) > 0:
                self.svp_figure.savefig(str(basepath) + "_SvsP.pdf", bbox_inches="tight")
            QMessageBox.information(self, "Export Successful", f"Saved plots with base: {basepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def save_curated_outputs(self, basepath: Path = None, save_bode_nyquist: bool = True, save_svsp: bool = False):
        """
        Save curated table (CSV) and plots to files.
        - Always saves curated CSV table
        - Optionally saves S vs P plot (default: False)
        - Optionally saves Bode and Nyquist if figures exist (default: True)
        """
        # pick a timestamped base path
        if basepath is None:
            if getattr(self, "save_mode", "timestamp") == "overwrite":
                basepath = self.export_dir / "hipoz_latest"
            elif self.save_mode == "rolling":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                basepath = self.export_dir / f"hipoz_{ts}"
            else:  # "timestamp"
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                basepath = self.export_dir / f"hipoz_{ts}"

        # 1) Save curated table
        try:
            # include 'associated' mask if you're tracking it; if not, this is a no-op
            df_to_save = self.data.copy()
            if hasattr(self, "associated_mask"):
                df_to_save["associated"] = self.associated_mask
            df_to_save.to_csv(str(basepath) + "_curated.csv", index=False)
            print(f"[HiPOZ] Saved curated CSV → {str(basepath)}_curated.csv")
        except Exception as e:
            print(f"[HiPOZ] Failed to save CSV: {e}")

        # 2) Save S vs P plot (optional)
        if save_svsp:
            try:
                # ensure S vs P is current
                if hasattr(self, "refresh_s_vs_p_plot"):
                    self.svp_figure.savefig(str(basepath) + "_SvsP.png", dpi=300, bbox_inches="tight")
                    self.svp_figure.savefig(str(basepath) + "_SvsP.pdf", bbox_inches="tight")
                    print(f"[HiPOZ] Saved S vs P → {str(basepath)}_SvsP.(png|pdf)")
            except Exception as e:
                print(f"[HiPOZ] Failed to save S vs P plot: {e}")

        if save_bode_nyquist:
            try:
                # save Bode if axes exist
                if hasattr(self, "bode_figure") and len(self.bode_figure.axes) > 0:
                    self.bode_figure.savefig(str(basepath) + "_Bode.png", dpi=300, bbox_inches="tight")
                    self.bode_figure.savefig(str(basepath) + "_Bode.pdf", bbox_inches="tight")
                    print(f"[HiPOZ] Saved Bode → {str(basepath)}_Bode.(png|pdf)")
            except Exception as e:
                print(f"[HiPOZ] Failed to save Bode plot: {e}")

            try:
                # save Nyquist if axes exist
                if hasattr(self, "nyquist_figure") and len(self.nyquist_figure.axes) > 0:
                    self.nyquist_figure.savefig(str(basepath) + "_Nyquist.png", dpi=300, bbox_inches="tight")
                    self.nyquist_figure.savefig(str(basepath) + "_Nyquist.pdf", bbox_inches="tight")
                    print(f"[HiPOZ] Saved Nyquist → {str(basepath)}_Nyquist.(png|pdf)")
            except Exception as e:
                print(f"[HiPOZ] Failed to save Nyquist plot: {e}")

        # optional cleanup for rolling mode
        if getattr(self, "save_mode", "timestamp") == "rolling":
            from glob import glob
            files = sorted(glob(str(self.export_dir / "hipoz_*_curated.csv")))
            # keep newest N, delete older
            if len(files) > self.rolling_keep:
                for old in files[0: len(files) - self.rolling_keep]:
                    stem = Path(old).with_suffix("").name.replace("_curated", "")
                    # remove the matching images/pdfs for that stem
                    for ext in ["_curated.csv", "_SvsP.png", "_SvsP.pdf",
                                "_Bode.png", "_Bode.pdf", "_Nyquist.png", "_Nyquist.pdf"]:
                        p = self.export_dir / f"{stem}{ext}"
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass

    def apply_calibration_config(self):
        """
        Apply calibration configuration automatically.

        Supports multiple calibration groups for bracketed measurements.
        """
        if not self.calib_config:
            return

        try:
            # Apply all calibration groups
            results = self.calib_config.apply_to_timeseries(self.timeseries)

            if not results:
                QMessageBox.warning(self, "Config Error",
                                  "No calibration groups found in config.")
                return

            # Get column indices
            col_Z = self.data.columns.get_loc('Z (Ohm)')
            col_dZ = self.data.columns.get_loc('Z± (Ohm)')
            col_S = self.data.columns.get_loc('S (S/m)')
            col_dS = self.data.columns.get_loc('S± (S/m)')
            col_cal = self.data.columns.get_loc('Calibration')

            # Mark all as not associated first
            self.associated_mask[:] = False
            # Clear calibration column
            self.data['Calibration'] = None

            # Track overall statistics
            total_standards = 0
            total_measurements = 0
            successful_groups = 0
            messages = []

            # Process each calibration group
            for result in results:
                if result['status'] != 'success':
                    messages.append(f"⚠️ {result['name']}: {result['message']}")
                    continue

                successful_groups += 1
                total_standards += len(result['standard_indices'])
                total_measurements += len(result['measurement_indices'])

                k_cell = result['cell_constant']
                k_cell_unc = result['cell_constant_unc']

                # Get standard filenames for display
                n_stds = len(result['standard_indices'])
                std_indices_str = ', '.join([str(i) for i in result['standard_indices']])

                # Populate standards with their known conductivity values
                for i, std_idx in enumerate(result['standard_indices']):
                    # Mark in Calibration column
                    self.data.iat[std_idx, col_cal] = f"{result['name']} [Standard]"

                    # Populate S (S/m) column with known conductivity
                    if 'standard_metadata' in result and i < len(result['standard_metadata']):
                        std_metadata = result['standard_metadata'][i]
                        if std_metadata.get('conductivity_Sm'):
                            self.data.iat[std_idx, col_S] = std_metadata['conductivity_Sm']
                            # Also update timeseries
                            try:
                                self.timeseries.conductivities_Sm[std_idx] = std_metadata['conductivity_Sm']
                            except Exception:
                                pass

                        # Update composition if provided
                        if std_metadata.get('comp'):
                            col_comp = self.data.columns.get_loc('Comp')
                            self.data.iat[std_idx, col_comp] = std_metadata['comp']

                # Apply to measurements in this group
                for i, idx in enumerate(result['measurement_indices']):
                    R_val = float(self.timeseries.Rcalc_ohm[idx])
                    dR_val = float(self.timeseries.uncertainties[idx])

                    if R_val > 0:
                        S_val = k_cell / R_val
                        dS_val = np.sqrt((S_val * dR_val / R_val) ** 2 +
                                       (k_cell * dR_val / R_val**2)**2)

                        # Update DataFrame
                        self.data.iat[idx, col_S] = S_val
                        self.data.iat[idx, col_dS] = dS_val

                        # Update timeseries
                        try:
                            self.timeseries.conductivities_Sm[idx] = S_val
                        except Exception:
                            pass

                        # Mark as associated
                        self.associated_mask[idx] = True

                    # Update Calibration column to show association with standards
                    self.data.iat[idx, col_cal] = f"{result['name']} [→ {n_stds} std{'s' if n_stds != 1 else ''}: {std_indices_str}]"

                    # Apply composition and concentration metadata if provided
                    if 'measurement_metadata' in result and i < len(result['measurement_metadata']):
                        metadata = result['measurement_metadata'][i]

                        # Update Comp column
                        if metadata.get('comp'):
                            col_comp = self.data.columns.get_loc('Comp')
                            self.data.iat[idx, col_comp] = metadata['comp']
                            try:
                                self.timeseries.comp[idx] = metadata['comp']
                            except Exception:
                                pass

                        # Update w (ppt) column - prefer ppt, fall back to molal
                        w_value = metadata.get('w_ppt')
                        if w_value is not None:
                            col_w = self.data.columns.get_loc('w (ppt)')
                            self.data.iat[idx, col_w] = w_value
                            try:
                                self.timeseries.ws_ppt[idx] = w_value
                            except Exception:
                                pass
                        elif metadata.get('w_molal') is not None:
                            # Convert molal to ppt if needed (for now just store molal value)
                            # TODO: Add conversion using PlanetProfile functions
                            col_w = self.data.columns.get_loc('w (ppt)')
                            self.data.iat[idx, col_w] = metadata['w_molal']
                            log.info(f"  Note: w_molal={metadata['w_molal']} stored (conversion to ppt not yet implemented)")
                            try:
                                self.timeseries.ws_ppt[idx] = metadata['w_molal']
                            except Exception:
                                pass

                messages.append(f"✓ {result['name']}: {result['message']}")

            # Refresh displays
            self.refresh_table()

            # Add visual indicators for pre-configured standards and measurements
            self._highlight_preconfigured_rows(results)

            self.refresh_s_vs_p_plot()
            self.save_curated_outputs()

            # Save GUI state to config file (creates both CSV and JSON)
            self.save_gui_state_to_config()

            # Build summary message
            summary = "\n".join(messages)

            if successful_groups > 0:
                QMessageBox.information(
                    self,
                    "Auto-Calibration Applied",
                    f"Applied {successful_groups} calibration group(s):\n\n"
                    f"{summary}\n\n"
                    f"Total: {total_standards} standards, {total_measurements} measurements\n"
                    f"Results saved to hipoz_exports/"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Calibration Failed",
                    f"No calibration groups succeeded:\n\n{summary}"
                )

        except Exception as e:
            import traceback
            QMessageBox.critical(
                self,
                "Auto-Calibration Error",
                f"Failed to apply calibration config:\n{str(e)}\n\n"
                f"You can still use manual calibration via GUI buttons."
            )
            traceback.print_exc()

    def _highlight_preconfigured_rows(self, results):
        """
        Add visual indicators to table rows that were pre-configured in CSV/JSON.

        Standards: Light blue background
        Measurements: Light green background
        """
        # Define colors
        standard_color = QColor(173, 216, 230)  # Light blue
        measurement_color = QColor(144, 238, 144)  # Light green

        for result in results:
            if result['status'] != 'success':
                continue

            # Highlight standards
            for idx in result['standard_indices']:
                for col in range(self.table.columnCount()):
                    item = self.table.item(idx, col)
                    if item:
                        item.setBackground(standard_color)
                        # Add tooltip showing this is a pre-configured standard
                        if col == 0:  # Only set tooltip on first column to avoid repetition
                            current_tooltip = item.toolTip()
                            if current_tooltip:
                                item.setToolTip(f"{current_tooltip}\n[Standard: {result['name']}]")
                            else:
                                item.setToolTip(f"Standard: {result['name']}")

            # Highlight measurements
            for idx in result['measurement_indices']:
                for col in range(self.table.columnCount()):
                    item = self.table.item(idx, col)
                    if item:
                        item.setBackground(measurement_color)
                        # Add tooltip showing this is a pre-configured measurement
                        if col == 0:  # Only set tooltip on first column
                            current_tooltip = item.toolTip()
                            if current_tooltip:
                                item.setToolTip(f"{current_tooltip}\n[Measurement: {result['name']}]")
                            else:
                                item.setToolTip(f"Measurement: {result['name']}")

    def check_config_file_locations(self):
        """
        Validate that zAnalysis<date>.json files exist in the correct data/<date>/ directories.

        This follows the naming convention: one zAnalysis<date>.json file per day,
        stored in the corresponding data/<date>/ directory.

        Auto-creates missing config files to store GUI analysis progress.

        Displays status in the status bar and logs findings.
        """
        # Extract dates from filenames
        dates_found = set()
        filename_by_date = {}  # Track filenames for each date

        for filename in self.timeseries.filenames:
            # Parse date from filename (format: Default_YYYYMMDD_... or similar patterns)
            # Try YYYYMMDD format first
            match = re.search(r'_(\d{8})_', filename)
            if match:
                date = match.group(1)
                dates_found.add(date)
                if date not in filename_by_date:
                    filename_by_date[date] = []
                filename_by_date[date].append(Path(filename).name)
            else:
                # Try other date formats like MM.DD.YY
                match = re.search(r'(\d{1,2})[._](\d{1,2})[._](\d{2,4})', filename)
                if match:
                    month, day, year = match.groups()
                    # Convert to YYYYMMDD format
                    if len(year) == 2:
                        year = '20' + year
                    date_str = f"{year}{month.zfill(2)}{day.zfill(2)}"
                    dates_found.add(date_str)
                    if date_str not in filename_by_date:
                        filename_by_date[date_str] = []
                    filename_by_date[date_str].append(Path(filename).name)

        if not dates_found:
            log.warning("Could not extract dates from filenames for config file validation")
            self.status_bar.showMessage("⚠️ No dates found in data filenames")
            return

        # Check each date for config files
        found_configs = []
        created_configs = []
        misplaced_configs = []

        for date in sorted(dates_found):
            # Check for both CSV and JSON (CSV preferred for Excel users)
            csv_path = Path('data') / date / f'zAnalysis{date}.csv'
            json_path = Path('data') / date / f'zAnalysis{date}.json'

            if csv_path.exists():
                expected_path = csv_path
                self.config_file_paths[date] = expected_path
                found_configs.append(date)
                log.info(f"✓ Found CSV config: {expected_path}")
            elif json_path.exists():
                expected_path = json_path
                self.config_file_paths[date] = expected_path
                found_configs.append(date)
                log.info(f"✓ Found JSON config: {expected_path}")
            else:
                # No config exists - check for misplaced files
                expected_path = csv_path  # Default to CSV for new files
                self.config_file_paths[date] = expected_path

                # Check multiple possible wrong locations
                wrong_csv = Path(f'zAnalysis{date}.csv')
                wrong_json = Path(f'zAnalysis{date}.json')

                if wrong_csv.exists():
                    misplaced_configs.append((date, wrong_csv, expected_path))
                    log.warning(f"⚠️ Config exists at wrong location: {wrong_csv}")
                    log.warning(f"  → Should be moved to: {expected_path}")
                elif wrong_json.exists():
                    misplaced_configs.append((date, wrong_json, csv_path))
                    log.warning(f"⚠️ Config exists at wrong location: {wrong_json}")
                    log.warning(f"  → Should be moved to: {csv_path} (or convert to CSV)")
                else:
                    # Auto-create CSV config file (Excel-friendly)
                    self.create_empty_config_file(date, expected_path, filename_by_date.get(date, []))
                    created_configs.append(date)
                    log.info(f"✓ Created new CSV config: {expected_path}")

        # Update status bar with summary
        total_found = len(found_configs) + len(created_configs)
        if total_found == len(dates_found):
            status_msg = f"✓ Config files ready ({len(found_configs)} found, {len(created_configs)} created)"
            self.status_bar.showMessage(status_msg)
            log.info(status_msg)
        elif found_configs or created_configs:
            status_msg = f"Config files: {len(found_configs)} found, {len(created_configs)} created, {len(misplaced_configs)} misplaced"
            self.status_bar.showMessage(status_msg)
            log.warning(status_msg)
        else:
            status_msg = f"⚠️ Issue with config files for {len(dates_found)} date(s)"
            self.status_bar.showMessage(status_msg)
            log.warning(status_msg)

        # Show detailed dialog if there are issues
        if created_configs or misplaced_configs:
            self.show_config_status_dialog(dates_found, found_configs, created_configs, misplaced_configs)

    def create_empty_config_file(self, date, file_path, filenames):
        """
        Create an empty zAnalysis<date> config file template.
        Defaults to CSV format (Excel-friendly) but supports JSON.

        Args:
            date: Date string (YYYYMMDD format)
            file_path: Path where to create the file
            filenames: List of filenames for this date
        """
        import json
        import csv as csv_module

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect format from extension
        is_csv = file_path.suffix.lower() == '.csv'

        try:
            if is_csv:
                # Create CSV format (Excel-friendly)
                with open(file_path, 'w', newline='') as f:
                    writer = csv_module.writer(f)
                    # Header with P_MPa, T_K, and exclude columns
                    writer.writerow(['group_name', 'filename', 'P_MPa', 'T_K', 'type', 'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes'])

                    # Parse P and T from filenames
                    def parse_p_t(filename):
                        match = re.search(r'P_(\d+)_T_(\d+)', filename)
                        if match:
                            return match.group(1), match.group(2)
                        return '', ''

                    # Separate KCl (likely standards) from others
                    for filename in filenames:
                        p_val, t_val = parse_p_t(filename)
                        if 'KCl' in filename or 'kcl' in filename.lower():
                            writer.writerow(['Group 1', filename, p_val, t_val, 'standard', '', 'KCl', '', '', '', 'Add conductivity_Sm (e.g., 0.0084 for 84 µS/cm)'])
                        else:
                            writer.writerow(['Group 1', filename, p_val, t_val, 'measurement', '', '', '', '', '', 'Add comp and w_ppt'])

                log.info(f"Created empty CSV config file: {file_path}")
                log.info("  Open in Excel to edit. Mark exclude='x' to skip files.")
            else:
                # Create JSON format
                config = {
                    "description": f"Analysis configuration for {date}",
                    "date": date,
                    "calibrations": [
                        {
                            "name": "Default group",
                            "standards": [],
                            "measurements": [{"filename": f} for f in filenames]
                        }
                    ],
                    "notes": [
                        "Auto-generated by GUI - edit to specify standards and composition",
                        "Move KCl files from measurements to standards and add conductivity_Sm",
                        "Add comp (e.g., 'NaCl') and w_ppt or w_molal for all measurements",
                        "Add 'exclude': true to skip files"
                    ]
                }

                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)

                log.info(f"Created empty JSON config file: {file_path}")

        except Exception as e:
            log.error(f"Failed to create config file {file_path}: {e}")

    def save_gui_state_to_config(self):
        """
        Save current GUI state (standards, measurements, conductivities) back to zAnalysis files.

        This persists the user's calibration work from the GUI back to the config files,
        allowing the analysis to be reproduced by re-running with the same config.

        Supports both CSV and JSON formats - saves in the same format as was loaded.
        """
        import json
        import csv as csv_module

        # Determine which config files to save to
        save_paths = {}

        # First priority: if we loaded a config via calib_config, save to that path
        if self.calib_config and hasattr(self.calib_config, 'config_path') and self.calib_config.config_path:
            config_path = Path(self.calib_config.config_path)
            # Extract date from the config path
            match = re.search(r'zAnalysis(\d{8})', str(config_path))
            if match:
                date = match.group(1)
                save_paths[date] = config_path
                log.info(f"Will save GUI state to loaded config: {config_path}")

        # Also save to any config file paths we discovered during initialization
        if self.config_file_paths:
            for date, path in self.config_file_paths.items():
                if date not in save_paths:  # Don't duplicate if already in save_paths
                    save_paths[date] = path

        if not save_paths:
            log.warning("No config file paths to save to")
            return

        # Group data by date
        data_by_date = {}
        for idx, filename in enumerate(self.timeseries.filenames):
            # Extract date from filename
            match = re.search(r'_(\d{8})_', filename)
            if match:
                date = match.group(1)
                if date not in data_by_date:
                    data_by_date[date] = {
                        'standards': [],
                        'measurements': []
                    }

                filename_only = Path(filename).name

                # Check if this is a standard (has been used for calibration)
                # For now, we'll mark all rows with S (S/m) values as potential measurements
                # and track standards separately via the mark_as_standard flow
                # This is a simplified approach - could be enhanced

                row_data = {
                    'filename': filename_only
                }

                # Add composition, concentration, and notes if available
                if idx < len(self.data):
                    comp = self.data.iat[idx, self.data.columns.get_loc('Comp')]
                    w_ppt = self.data.iat[idx, self.data.columns.get_loc('w (ppt)')]
                    S_Sm = self.data.iat[idx, self.data.columns.get_loc('S (S/m)')]

                    if comp and pd.notna(comp):
                        row_data['comp'] = comp
                    if w_ppt is not None and pd.notna(w_ppt):
                        row_data['w_ppt'] = float(w_ppt)

                    # If S (S/m) is set and this is associated, treat as measurement
                    # Otherwise, it could be a standard (would need conductivity_Sm)
                    if self.associated_mask[idx]:
                        data_by_date[date]['measurements'].append(row_data)
                    else:
                        # Could be a standard or unprocessed measurement
                        if S_Sm is not None and pd.notna(S_Sm):
                            # Has known conductivity - treat as standard
                            row_data['conductivity_Sm'] = float(S_Sm)
                            data_by_date[date]['standards'].append(row_data)
                        else:
                            # Unprocessed measurement
                            data_by_date[date]['measurements'].append(row_data)

        # Write to each date's config file (CSV or JSON format)
        for date, file_path in save_paths.items():
            if date not in data_by_date:
                continue

            try:
                # Detect format from extension
                is_csv = file_path.suffix.lower() == '.csv'
                save_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save to primary format (CSV or JSON based on input)
                if is_csv:
                    self._save_to_csv(file_path, data_by_date[date], date, save_timestamp)
                    log.info(f"Saved GUI state to CSV: {file_path}")

                    # Also save matching JSON for harmonization
                    json_path = file_path.with_suffix('.json')
                    self._save_to_json(json_path, data_by_date[date], date, save_timestamp)
                    log.info(f"Created matching JSON: {json_path}")
                else:
                    self._save_to_json(file_path, data_by_date[date], date, save_timestamp)
                    log.info(f"Saved GUI state to JSON: {file_path}")

                    # Also save matching CSV for harmonization
                    csv_path = file_path.with_suffix('.csv')
                    self._save_to_csv(csv_path, data_by_date[date], date, save_timestamp)
                    log.info(f"Created matching CSV: {csv_path}")

            except Exception as e:
                log.error(f"Failed to save GUI state to {file_path}: {e}")

    def _save_to_csv(self, file_path: Path, data: dict, date: str, timestamp: str):
        """Save GUI state to CSV format."""
        import csv as csv_module

        # Parse P and T from filenames helper
        def parse_p_t(filename):
            match = re.search(r'P_(\d+)_T_(\d+)', filename)
            if match:
                return match.group(1), match.group(2)
            return '', ''

        # Read existing CSV to preserve structure and notes
        existing_rows = []
        existing_groups = set()
        existing_notes = {}  # Map filename -> notes
        if file_path.exists():
            with open(file_path, 'r') as f:
                reader = csv_module.DictReader(f)
                existing_rows = list(reader)
                for row in existing_rows:
                    existing_groups.add(row.get('group_name', 'Group 1'))
                    # Preserve existing notes
                    if row.get('filename') and row.get('notes'):
                        existing_notes[row['filename']] = row['notes']

        # Write updated CSV with P_MPa and T_K columns
        with open(file_path, 'w', newline='') as f:
            fieldnames = ['group_name', 'filename', 'P_MPa', 'T_K', 'type', 'conductivity_Sm', 'comp', 'w_ppt', 'w_molal', 'exclude', 'notes']
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            group_name = list(existing_groups)[0] if existing_groups else 'Group 1'

            # Write standards
            for std in data['standards']:
                p_val, t_val = parse_p_t(std['filename'])
                # Preserve existing notes if available
                existing_note = existing_notes.get(std['filename'], '')
                row = {
                    'group_name': group_name,
                    'filename': std['filename'],
                    'P_MPa': p_val,
                    'T_K': t_val,
                    'type': 'standard',
                    'conductivity_Sm': std.get('conductivity_Sm', ''),
                    'comp': std.get('comp', ''),
                    'w_ppt': std.get('w_ppt', ''),
                    'w_molal': std.get('w_molal', ''),
                    'exclude': '',
                    'notes': existing_note  # Preserve user's notes
                }
                writer.writerow(row)

            # Write measurements
            for meas in data['measurements']:
                p_val, t_val = parse_p_t(meas['filename'])
                # Preserve existing notes if available
                existing_note = existing_notes.get(meas['filename'], '')
                row = {
                    'group_name': group_name,
                    'filename': meas['filename'],
                    'P_MPa': p_val,
                    'T_K': t_val,
                    'type': 'measurement',
                    'conductivity_Sm': '',
                    'comp': meas.get('comp', ''),
                    'w_ppt': meas.get('w_ppt', ''),
                    'w_molal': meas.get('w_molal', ''),
                    'exclude': '',
                    'notes': existing_note  # Preserve user's notes
                }
                writer.writerow(row)

    def _save_to_json(self, file_path: Path, data: dict, date: str, timestamp: str):
        """Save GUI state to JSON format."""
        import json

        # Load existing config or create new
        if file_path.exists():
            with open(file_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "description": f"Analysis configuration for {date}",
                "date": date,
                "calibrations": [],
                "notes": []
            }

        # Update the default calibration group
        if not config.get('calibrations'):
            config['calibrations'] = []

        # Find or create default group
        if len(config['calibrations']) == 0:
            config['calibrations'].append({
                "name": "Default group",
                "standards": [],
                "measurements": []
            })

        # Update first group with current state
        config['calibrations'][0]['standards'] = data['standards']
        config['calibrations'][0]['measurements'] = data['measurements']

        # Add note about GUI save
        if 'notes' not in config:
            config['notes'] = []
        config['notes'].append(f"Updated from GUI at {timestamp}")

        # Write back
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

    def show_config_status_dialog(self, all_dates, found, created, misplaced):
        """
        Show a dialog with detailed status of config files.

        Args:
            all_dates: Set of all dates found in data
            found: List of dates with configs in correct location
            created: List of dates where configs were auto-created
            misplaced: List of (date, wrong_path, correct_path) tuples
        """
        msg_parts = []

        msg_parts.append("Config File Status Check")
        msg_parts.append("=" * 50)
        msg_parts.append("")
        msg_parts.append(f"Dates found in data: {', '.join(sorted(all_dates))}")
        msg_parts.append("")

        if found:
            msg_parts.append(f"✓ Found ({len(found)}):")
            for date in sorted(found):
                msg_parts.append(f"  • zAnalysis{date}.json in data/{date}/")
            msg_parts.append("")

        if created:
            msg_parts.append(f"✓ Created ({len(created)}):")
            for date in sorted(created):
                msg_parts.append(f"  • zAnalysis{date}.json in data/{date}/")
            msg_parts.append("")
            msg_parts.append("These files will store your GUI analysis progress.")
            msg_parts.append("Edit them to specify:")
            msg_parts.append("  • Move KCl files to standards array")
            msg_parts.append("  • Add conductivity_Sm values for standards")
            msg_parts.append("  • Add comp and w_ppt/w_molal for measurements")
            msg_parts.append("")

        if misplaced:
            msg_parts.append(f"⚠️ Misplaced ({len(misplaced)}):")
            for date, wrong, correct in misplaced:
                msg_parts.append(f"  • {wrong}")
                msg_parts.append(f"    → Move to: {correct}")
            msg_parts.append("")
            msg_parts.append("To fix: mv <file> <correct_location>")
            msg_parts.append("")

        msg_text = "\n".join(msg_parts)

        # Create info dialog
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle("Config File Location Check")
        dialog.setText("Config files following zAnalysis<date>.json convention")
        dialog.setInformativeText(
            f"Naming convention:\n"
            f"• One zAnalysis<date>.json per day\n"
            f"• Stored in data/<date>/ directory\n"
            f"• Auto-created if missing to save GUI progress"
        )
        dialog.setDetailedText(msg_text)
        dialog.setStandardButtons(QMessageBox.Ok)

        # Show if files were created or misplaced
        if created or misplaced:
            dialog.exec_()


def main():
    app = QApplication(sys.argv)
    window = DataSelector(timeseries)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()