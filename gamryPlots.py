import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import logging
from gamryTools import ResistorData, Solution

# Assign logger
log = logging.getLogger('HiPOZ')

# Configure LaTeX rendering (with fallback if LaTeX not available)
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{stix}\usepackage{siunitx}\usepackage{upgreek}\usepackage[version=4]{mhchem}\sisetup{round-mode=places,scientific-notation=true,round-precision=2}'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'STIXGeneral'
except Exception as e:
    log.warning(f'LaTeX rendering not available: {e}. Falling back to default rendering.')
    plt.rcParams['text.usetex'] = False
plot_air = False
tfmt = '%b%d%H%M%S'
uScm = r'\mu{S\,cm^{-1}}'
MS_data = 'o'
LS_fit = '-'

def get_unique(handles, labels, paired=True):
    out_labels = np.unique(labels)
    if not paired:
        data_labels = np.where([not 'fit' in label for label in out_labels])[0]
        fit_labels = np.where(['fit' in label for label in out_labels])[0]
        out_labels = np.concatenate((out_labels[data_labels], out_labels[fit_labels]))
    i_unique = np.array([next(i for i, label in enumerate(labels) if label == u_label) for u_label in out_labels])
    out_handles = [handles[i] for i in i_unique]
    out_labels = [labels[i] for i in i_unique]

    return out_handles, out_labels


def add_ticks_x(r_ticks, line_list, ax):
    default_ticks = ax.get_xticks()
    default_tick_labels = [f'$10^{{{np.log10(tick):.0f}}}$' for tick in default_ticks]
    r_tick_labels = [f'$\\num{{{r_tick}}}$' for r_tick in r_ticks]
    n_default_ticks = np.size(default_ticks)
    all_ticks = np.concatenate((default_ticks, r_ticks))
    all_tick_labels = np.concatenate((default_tick_labels, r_tick_labels))

    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_tick_labels)
    line_colors = [line.get_color() for line in line_list]
    #line_styles = [line.set_linestyle('--') for line in line_list[::2]]
    plt.setp(ax.xaxis.get_ticklabels()[n_default_ticks:], rotation='vertical')
    [plt.setp(tick, color=color) for tick, color in zip(ax.xaxis.get_ticklabels()[n_default_ticks:], line_colors)]


def add_ticks_y(r_ticks, line_list, ax):
    default_ticks = ax.get_yticks()
    default_tick_labels = [f'$10^{{{np.log10(tick):.0f}}}$' for tick in default_ticks]
    r_tick_labels = [f'$\\num{{{r_tick}}}$' for r_tick in r_ticks]
    n_default_ticks = np.size(default_ticks)
    all_ticks = np.concatenate((default_ticks, r_ticks))
    all_tick_labels = np.concatenate((default_tick_labels, r_tick_labels))

    ax.set_yticks(all_ticks)
    ax.set_yticklabels(all_tick_labels)
    line_colors = [line.get_color() for line in line_list]
    #line_styles = [line.set_linestyle('--') for line in line_list[::2]]
    plt.setp(ax.yaxis.get_ticklabels()[n_default_ticks:], rotation='horizontal')
    [plt.setp(tick, color=color) for tick, color in zip(ax.yaxis.get_ticklabels()[n_default_ticks:], line_colors)]


def plot_z(sols, fig_size, out_fig_name, xtn, r_ticks, add=None, leg_pairs=True):
    fig = plt.figure(figsize=fig_size)
    grid = GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlabel(r'$\mathrm{Re}\{Z\}$ ($\Omega$)')
    ax.set_ylabel(r'$-\mathrm{Im}\{Z\}$ ($\Omega$)')
    ax.set_title(r'Calibration solution complex impedance')
    ax.set_xscale('log')
    ax.set_yscale('log')

    dot_list = [ax.scatter(np.real(sol.Z_ohm), -np.imag(sol.Z_ohm), marker=MS_data, label=f'{sol.legLabel} data', color=sol.color) for sol in sols]
    line_list = [ax.plot(np.real(sol.Z_ohm), -np.imag(sol.Z_ohm), ls=LS_fit, label=f'{sol.legLabel} fit', color=sol.fitColor)[0] for sol in sols]
    if r_ticks is not None:
        add_ticks_x(r_ticks, line_list, ax)

    all_handles, all_labels = ax.get_legend_handles_labels()
    handles, labels = get_unique(all_handles, all_labels, paired=leg_pairs)
    ax.legend(handles, labels, title=r'$\sigma_\mathrm{std}$ ($\mathrm{S/m}$)')
    plt.tight_layout()
    if add is None:
        add_bit = ''
    else:
        add_bit = add
    out_fname = f'{out_fig_name}Z{add_bit}.{xtn}'
    fig.savefig(out_fname, format=xtn, dpi=200)
    log.info(f'Gamry calibration plot saved to file: {out_fname}')
    plt.close()


def plot_y(sols, fig_size, out_fig_name, xtn, r_ticks, add=None, leg_pairs=True):
    fig = plt.figure(figsize=fig_size)
    grid = GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlabel(r'$\mathrm{Re}\{Y\}$ ($\mho$)')
    ax.set_ylabel(r'$-\mathrm{Im}\{Y\}$ ($\mho$)')
    ax.set_title(r'Calibration solution complex conductance')
    ax.set_xscale('log')
    ax.set_yscale('log')

    dotList = [ax.scatter(1/np.real(sol.Z_ohm), -1/np.imag(sol.Z_ohm), marker=MS_data, label=f'{sol.legLabel} data', color=sol.color) for sol in sols]
    lineList = [ax.plot(1/np.real(sol.Zfit_ohm), -1/np.imag(sol.Zfit_ohm), ls=LS_fit, label=f'{sol.legLabel} fit', color=sol.fitColor)[0] for sol in sols]
    if r_ticks is not None:
        add_ticks_x(r_ticks, line_list, ax)

    all_handles, all_labels = ax.get_legend_handles_labels()
    handles, labels = get_unique(all_handles, all_labels, paired=leg_pairs)
    ax.legend(handles, labels, title=r'$\sigma_\mathrm{std}$ ($\mathrm{S/m}$)')
    plt.tight_layout()
    if add is None:
        add_bit = ''
    else:
        add_bit = add
    out_fname = f'{out_fig_name}Y{add_bit}.{xtn}'
    fig.savefig(out_fname, format=xtn, dpi=200)
    log.info(f'Gamry calibration plot saved to file: {out_fname}')
    plt.close()

    return


def plot_zvsf(sols, fig_size, out_fig_name, xtn, r_ticks, add=None, leg_pairs=True):
    fig = plt.figure(figsize=fig_size)
    grid = GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlabel(r'Frequency $f$ ($\mathrm{Hz}$)')
    ax.set_ylabel(r'Impedance $|Z|$ ($\Omega$)')
    ax.set_title(r'Electrochemical impedance spectra')
    ax.set_xscale('log')
    ax.set_yscale('log')

    dotList = [ax.scatter(sol.f_Hz, np.abs(sol.Z_ohm), marker=MS_data, label=f'{sol.legLabel} data', color=sol.color) for sol in sols]
    lineList = [ax.plot(sol.f_Hz, np.abs(sol.Zfit_ohm), ls=LS_fit, label=f'{sol.legLabel} fit', color=sol.fitColor)[0] for sol in sols]
    if r_ticks is not None:
        add_ticks_y(r_ticks, line_list, ax)

    all_handles, all_labels = ax.get_legend_handles_labels()
    handles, labels = get_unique(all_handles, all_labels, paired=leg_pairs)
    ax.legend(handles, labels, title=r'$\sigma_\mathrm{std}$ ($\mathrm{S/m}$)')
    plt.tight_layout()
    if add is None:
        add_bit = ''
    else:
        add_bit = add
    out_fname = f'{out_fig_name}Zvsf{add_bit}.{xtn}'
    fig.savefig(out_fname, format=xtn, dpi=200)
    log.info(f'Gamry calibration plot saved to file: {out_fname}')
    plt.close()

    return


def plot_phasevsf(sols, fig_size, out_fig_name, xtn, add=None, leg_pairs=True):
    fig = plt.figure(figsize=fig_size)
    grid = GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlabel(r'Frequency $f$ ($\mathrm{Hz}$)')
    ax.set_ylabel(r'Phase ($^\circ$)')
    ax.set_title(r'Calibration solution phase spectrum')
    ax.set_xscale('log')

    dotList = [ax.scatter(sol.f_Hz, np.angle(sol.Z_ohm, deg=True), marker=MS_data, color=sol.color, label=f'{sol.legLabel} data') for sol in sols]
    lineList = [ax.plot(sol.f_Hz, np.angle(sol.Zfit_ohm, deg=True), ls=LS_fit, color=sol.fitColor, label=f'{sol.legLabel} fit')[0] for sol in sols]

    all_handles, all_labels = ax.get_legend_handles_labels()
    handles, labels = get_unique(all_handles, all_labels, paired=leg_pairs)
    ax.legend(handles, labels, title=r'$\sigma_\mathrm{std}$ ($\mathrm{S/m}$)')
    plt.tight_layout()
    if add is None:
        add_bit = ''
    else:
        add_bit = add
    out_fname = f'{out_fig_name}Phasevsf{add_bit}.{xtn}'
    fig.savefig(out_fname, format=xtn, dpi=200)
    log.info(f'Gamry calibration plot saved to file: {out_fname}')
    plt.close()

    return


def plot_condvsp(sols, fig_size, out_fig_name, xtn, add=None):
    fig = plt.figure(figsize=fig_size)
    grid = GridSpec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlabel(r'Pressure $P$ ($\mathrm{MPa}$)')
    ax.set_ylabel(r'Conductivity $\sigma$ ($\mathrm{S/m}$)')
    ax.set_title(r'Conductivity vs Pressure' + date)
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    Plist = [sol.P_MPa for sol in sols]
    Siglist = [sol.sigma_Sm * 1e-4 for sol in sols]
    for sol in sols:
        ax.plot(sol.P_MPa, sol.sigma_Sm, marker='o', markerfacecolor='g', markeredgecolor='k')
    # lineList = [ax.plot(sol.P_MPa, sol.sigma_Sm, label=f'{sol.legLabel} data', color=sol.color, marker='o')[0]  for sol in sols]
    # if Rticks is not None:
    #     AddTicksY(Rticks, lineList, ax)

    # ax.legend(title=r'$\sigma_\mathrm{std}$ ($\mathrm{S/m}$)')
    plt.tight_layout()
    if add is None:
        add_bit = ''
    else:
        add_bit = add
    out_fname = f'{out_fig_name}CondvsP{add_bit}.{xtn}'
    fig.savefig(out_fname, format=xtn, dpi=200)
    log.info(f'Cond vs P plot saved to file: {out_fname}')
    plt.close()

    return


def plot_zfit(sols, fig_size, xtn, out_fig_name=None, leg_pairs=True):
    for sol in sols:
        if isinstance(sol, ResistorData):
            # This is a resistor data, plot data and fit if available
            fig = plt.figure(figsize=fig_size)
            grid = GridSpec(1, 1)
            ax = fig.add_subplot(grid[0, 0])
            ax.grid()
            ax.set_axisbelow(True)
            ax.set_xlabel('Re($Z$)')
            ax.set_ylabel('$-$Im($Z$)')

            if sol.f_Hz is not None and len(sol.f_Hz) > 0:
                ax.scatter(np.real(sol.Z_ohm), -np.imag(sol.Z_ohm), marker=MS_data, label=f'{sol.legLabel} data',
                           color=sol.color)
            if sol.Zfit_ohm is not None and len(sol.Zfit_ohm) > 0:
                ax.plot(np.real(sol.Zfit_ohm), -np.imag(sol.Zfit_ohm), ls=LS_fit, label=f'{sol.legLabel} fit',
                        color=sol.fitColor)

            ax.set_xlim(left=0)

            all_handles, all_labels = ax.get_legend_handles_labels()
            handles, labels = get_unique(all_handles, all_labels, paired=leg_pairs)
            ax.legend(handles, labels)

            if out_fig_name is not None:
                tstr = sol.time.strftime(tfmt)  # temporary comment out CP for resistor testing
                thisOutFigName = f'{sol.lbl_uScm}uScm_{tstr}'
                ax.set_title(
                    f'Nyquist plot for ${sol.lbl_uScm}\,\mathrm{{\mu S/cm}}$ at {tstr}, $K_\mathrm{{cell}}={sol.Kcell_pm:.2f}\,\mathrm{{m^{{-1}}}}$')

                out_fname = f'{thisOutFigName}Nyquist.{xtn}'
                fig.savefig(out_fname, format=xtn, dpi=200)
                log.info(f'Nyquist plot saved to file: {out_fname}')
                plt.close()

        elif isinstance(sol, Solution):
            # This is a solution data, do not plot Nyquist plot for it
            pass

    return

def add_pt(ax, x, p, t):
    ax.plot(x, t, '-r', label='Temperature')
    # ax.plot(x, p, '-g', label='Pressure')
    # ax.set_ylabel('Temperature (°C) / Pressure (MPa)', color='red')
    ax.set_ylabel('Temperature (°C)', color='red')
    ax.tick_params(axis='y', labelcolor='red')

def plot_timeseries(timeseries, fig_size=None, out_fig_name=None, xtn=None, figure=None, interactive=False):

    # Create a figure with subplots
    if figure is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(26, 14), sharex=True)
    else:
        fig = figure
        ax1 = fig.add_subplot(211) # impedance values with errorbars
        ax2 = fig.add_subplot(212) # bottom plot for uncertainty

    # Plot impedance with uncertainties
    # Plot each data point with specific marker and color
    for ts, imp, unc, color, marker in zip(timeseries.timestamps, timeseries.Rcalc_ohm, timeseries.uncertainties, timeseries.colors, timeseries.markers):
        plot_element, caplines, barlinecols = ax1.errorbar(ts, imp, yerr=unc, fmt=marker, color=color, capsize=5)
        if interactive:
            plot_element.set_picker(5)  # 5 points tolerance

    # ax1.errorbar(timestamps, Rcalc_ohm, yerr=uncertainties, fmt='o', capsize=5, label='Impedance with Uncertainty')
    ax1.set_ylabel('Impedance (Ohm)')
    ax1.set_title('Impedance Measurement Over Time')
    # ax1.legend()
    ax1.grid(True)


    # Plot percent uncertainties
    # Plot each data point with specific marker and color
    for ts, imp, unc, color, marker in zip(timeseries.timestamps, timeseries.Rcalc_ohm, timeseries.percent_uncertainties, timeseries.colors, timeseries.markers):
        ax2.plot(ts, unc, marker=marker, color=color)
    # ax2.plot(timestamps, percent_uncertainties, 'o', label='Percent Uncertainty')
    ax2.set_xlabel('Time',labelpad=-20)
    ax2.set_ylabel('Percent Uncertainty (%)')
    ax2.set_title('Percent Uncertainty Over Time')
    # ax2.legend()
    ax2.grid(True)

    # ax3 = ax1.twinx()
    # addPT(ax3,timestamps,Ps,Ts)
    # ax4 = ax2.twinx()
    # addPT(ax4,timestamps,Ps,Ts)
    # Adjust subplot parameters manually if necessary

    # Adjust layout and display
    # Rotate and align x-axis labels
    # plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    # plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    # plt.tight_layout()  # Adjust layout to make room for label rotations
    # plt.subplots_adjust(bottom=0.2,left=0.2,right=0.2,top=0.2)  # Adjust margins if necessary
    plt.subplots_adjust(bottom=0.2, top=0.8, left=0.1)  # You can tweak these values

    for ax in ax1, ax2:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
        plt.gcf().autofmt_xdate()  # Rotate date labels to avoid overlap

    if interactive:
        return fig, ax1, ax2
    else:
        plt.show()




def plot_sigma(all_meas, fig_size, out_fig_name, xtn):
    colorstr = 'gbryk'
    iColor = 0

    fig = plt.figure(figsize=fig_size)
    grid = GridSpec(1, 1)
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    for ax in ax1, ax2:
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_ylabel('$\sigma$(S/m)')
        ax.set_yscale('log')
    ax1.set_xlabel('$T$(K)')
    ax2.set_xlabel('$P$(MPa)')

    allPs = np.zeros_like(allMeas)
    allTs = np.zeros_like(allMeas)
    allSigs = np.zeros_like(allMeas)
    for i,meas in enumerate(allMeas):
        allPs[i] = [thisMeas.P_MPa for thisMeas in meas]
        allTs[i] = [thisMeas.T_K for thisMeas in meas]
        allSigs[i] = [thisMeas.sigma_Sm for thisMeas in meas]

    for meas in allMeas:
        Plist = np.array([thisMeas.P_MPa for thisMeas in meas])
        # klist = [thisMeas. for thisMeas in meas]
        iSort  = np.argsort(Plist)
        Tlist = np.array([thisMeas.T_K for thisMeas in meas])
        Siglist = np.array([thisMeas.sigma_Sm for thisMeas in meas])
        # for thisMeas in meas:
        #     ax.plot(thisMeas.P_MPa,thisMeas.sigma_Sm, marker='o', markerfacecolor = colorstr[iColor], markeredgecolor  = 'k')
        for thisMeas in meas:
            # ax1.plot(thisMeas.T_K,thisMeas.sigma_Sm, marker='o', markerfacecolor = colorstr[iColor], markeredgecolor  = 'k')
            ax1.plot(thisMeas.T_K,thisMeas.sigma_Sm, marker='o', markerfacecolor = thisMeas.color, markeredgecolor  = 'k')
            ax2.plot(thisMeas.P_MPa,thisMeas.sigma_Sm, marker='o', markerfacecolor = thisMeas.color, markeredgecolor  = 'k')
        iColor += 1

    # ax.set_yscale('log')
    plt.show()
    # plt.tight_layout()
    out_fname = f'{out_fig_name}CondvsTP.{xtn}'
    fig.savefig(out_fname, format=xtn, dpi=500)
    print(f'Cond vs T plot saved to file: {out_fname}')
    plt.close()
