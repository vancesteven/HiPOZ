import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
                             QListWidget, QLabel, QMessageBox, QComboBox, QFileDialog,
                             QTableWidget, QTableWidgetItem, QTabWidget, QHBoxLayout)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from gamryPlots import PlotTimeseries

class DataSelector(QMainWindow):
    def __init__(self, timeseries):
        super(DataSelector,self).__init__()
        self.timeseries = timeseries  # This is the array of Solution objects supplied externally
        self.updating_table = False
        self.selected_points = []
        self.ax1 = []
        self.ax2 = []
        self.initUI()
        self.current_std = []
        self.current_std_uncertainty = []
        self.current_meas = []
    def initUI(self):
        self.setGeometry(200, 200, 1000, 800)
        self.setWindowTitle('Gamry Data')

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

        self.data = pd.DataFrame({
            'Time': self.timeseries.timestamps,
            'Comp': self.timeseries.comp,
            'w (ppt)': self.timeseries.ws_ppt,
            'T (K)': self.timeseries.Ts,
            'P (MPa)': self.timeseries.Ps,
            'Z (Ohm)': self.timeseries.Rcalc_ohm,
            'Z± (Ohm)': self.timeseries.percent_uncertainties,
            'S (S/m)': self.timeseries.conductivities_Sm,
            'S± (S/m)': self.timeseries.conductivities_unc_pct
        })
        self.associated_mask = np.zeros(len(self.data), dtype=bool)  # rows marked by Associate Measurements
        # Cast 'w (ppt)' and 'P (MPa)' to int
        # self.data['w (ppt)'] = self.data['w (ppt)'].astype(int)
        self.data['P (MPa)'] = self.data['P (MPa)'].astype(int)
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

        # default export directory
        self.export_dir = Path.cwd() / "hipoz_exports"
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
            self.figure, self.ax1, self.ax2 = PlotTimeseries(self.timeseries,Figure=self.figure,figSize= (26, 14), interactive=True)
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
                val = float(text)
            elif header in ['P (MPa)']:
                val = int(float(text))
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
        try:
            if header == 'Z (Ohm)':
                self.timeseries.Rcalc_ohm[row] = float(val)
            elif header == 'S (S/m)':
                self.timeseries.conductivities_Sm[row] = float(val)
            elif header == 'P (MPa)':
                self.timeseries.Ps[row] = int(val)
            elif header == 'T (K)':
                self.timeseries.Ts[row] = float(val)
            elif header == 'w (ppt)':
                self.timeseries.ws_ppt[row] = float(val)
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
                dz_val = float(self.timeseries.dz_ohm[row])

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
                    if isinstance(val, pd.Timestamp):
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

    def save_curated_outputs(self, basepath: Path = None, save_bode_nyquist: bool = True):
        """
        Save curated table (CSV) and plots to files.
        - Always saves S vs P plot if available.
        - Optionally saves Bode and Nyquist if figures exist.
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

        # 2) Save plots
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


def main():
    app = QApplication(sys.argv)
    window = DataSelector(timeseries)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()