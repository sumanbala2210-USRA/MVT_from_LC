"""
trigger_process.py
MVT analysis workflow for Fermi GBM triggers.
"""

import os
import sys
import yaml
import glob
import argparse
import functools
import concurrent.futures
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from tqdm import tqdm

# --- GDT & Astro Imports ---
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.finders import TriggerFtp, TriggerFinder
from gdt.core.binning.unbinned import bin_by_time
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial
from gdt.missions.fermi.gbm.collection import GbmDetectorCollection
from gdt.missions.fermi.time import *
from gdt.core.phaii import Phaii
from gdt.missions.fermi.gbm.trigdat import Trigdat
from gdt.core.plot.lightcurve import Lightcurve

# --- Local Module Imports ---
from SIM_lib import run_mvt_in_subprocess, send_email
from poly_find import find_best_poly_order
from TTE_SIM_v2 import (
    _calculate_multi_timescale_snr, 
    analysis_mvt_time_resolved_results_to_dataframe, 
    create_final_GBM_plot_with_MVT, 
    analysis_mvt_results_to_dataframe, 
    write_yaml,
    print_nested_dict
)

# --- Global Configurations ---
CONFIG_FILE = 'config_MVT_fermi.yaml'
MVT_CONFIG_FILE = 'simulations_ALL.yaml'
MAX_WORKERS = max(1, os.cpu_count() - 2)


# ==========================================
# 1. Aesthetics & Styling
# ==========================================

def setup_publication_style():
    """Sets Matplotlib parameters for high-quality, publication-ready plots."""
    plt.style.use('seaborn-v0_8-paper')
    params = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'], 
        'font.size': 14,               
        'axes.labelsize': 16,          
        'axes.titlesize': 16,          
        'xtick.labelsize': 14,         
        'ytick.labelsize': 14,         
        'legend.fontsize': 14,         
        'figure.titlesize': 18,        
        'lines.linewidth': 1.5,
        'xtick.direction': 'in',       
        'ytick.direction': 'in',
        'xtick.top': True,             
        'ytick.right': True,
        'savefig.dpi': 300,            
        'savefig.bbox': 'tight',       
    }
    plt.rcParams.update(params)

# ==========================================
# 2. Data Retrieval & Parsing 
# ==========================================

def download_data(trigger_number, path, flags={'tte':False, 'rsp':False, 'cat':False, 'trigdat':False}):
    """Downloads necessary GRB data files from AWS."""
    try:
        trigger_ftp = TriggerFinder(trigger_number, protocol='AWS')
    except Exception as e:
        print(f"Failed to initialize TriggerFinder for {trigger_number}: {e}")
        exit(1)
        
    if flags.get('tte', False): trigger_ftp.get_tte(download_dir=path)
    if flags.get('rsp', False): trigger_ftp.get_rsp2(download_dir=path)
    if flags.get('cat', False): trigger_ftp.get_cat_files(download_dir=path)
    if flags.get('trigdat', False): trigger_ftp.get_trigdat(download_dir=path)
    
def trigger_file_list(trigger_dir, file_type, trigger_number, nai_dets=None, bgo_flag=False):
    """Generates lists of file absolute paths and names based on constraints."""
    file_abs_path_list = []
    if file_type in ['trigdat', 'tcat', 'bcat']:
        bgo_flag = True
        
    if nai_dets == 'all' or nai_dets is None:
        pattern = trigger_dir / f"glg_{file_type}_{'*' if bgo_flag else 'n*'}_bn{trigger_number}*.fit"
        file_abs_path_list = glob.glob(str(pattern))
    else:
        for det in nai_dets:
            pattern = trigger_dir / f"glg_{file_type}_{det}_bn{trigger_number}*.fit"
            file_abs_path_list.extend(glob.glob(str(pattern)))

    file_name_list = [os.path.basename(path) for path in file_abs_path_list]
    return sorted(file_abs_path_list), sorted(file_name_list)

def get_GRB_par(trigger_number, trigger_directory):
    """Extracts duration and flux parameters from BCAT files."""
    try:
        bcat_down_list_path, _ = trigger_file_list(trigger_directory, "bcat", trigger_number)
        bcat_hdu = fits.open(bcat_down_list_path[-1])
        header = bcat_hdu[0].header
        T0 = round(float(header['T90START']), 4)
        T90 = round(float(header['T90']), 4)
        T50 = round(float(header['T50']), 4)
        PF64 = round(float(header['PF64']), 4)
        PFLX = round(float(header['PFLX']), 4)
        FLU = round(float(header['FLU'])*1e6, 4)
        bcat_hdu.close()
        return T0, T90, T50, PF64, PFLX, FLU
    except Exception as e:
        print(f"Failed to get T90 for {trigger_number}: {e}")
        return None, None, None, None, None, None

def check_GBM_status(tig_time):
    """Checks against gbm_status.csv to filter out bad detectors during trigger window."""
    problem_detectors = []
    df = pd.read_csv('gbm_status.csv')
    for i in range(0,len(df.TSTART)):
        bad_time_start = Time(df.TSTART[i], format='iso')
        bad_time_stop = Time(df.TSTOP[i], format='iso')
        
        if bad_time_start.fermi < tig_time.fermi < bad_time_stop.fermi:
            print(f"Trigger Within bad time {df.TSTART[i]} & {df.TSTOP[i]}")
            print(f"Problem with {df.Detectors[i]} detector(s): {df.Comment[i]}\n")
            if 'all' in df.Detectors[i]:
                problem_detectors = [det.name for det in GbmDetectors]
            else:
                problem_detectors = [word.lower() for word in df.Detectors[i].split()] 
    return problem_detectors

def get_dets_list(trigger_number, trigger_directory):
    """Identifies the best detectors observing the source based on pointing angles."""
    try:
        tcat_down_list_path, _ = trigger_file_list(trigger_directory, "tcat", trigger_number)
        tcat_hdu = fits.open(tcat_down_list_path[-1])
        source_coord = SkyCoord(tcat_hdu[0].header['RA_OBJ'], tcat_hdu[0].header['DEC_OBJ'], frame='icrs', unit='deg')
        tcat_hdu.close()
    except Exception as e:
        print(f"Failed to get source coordinates for {trigger_number}: {e}")

    try:
        trigdat_down_list_path, _ = trigger_file_list(trigger_directory, "trigdat", trigger_number)
        trigdat = Trigdat.open(trigdat_down_list_path[-1])
        tig_time = Time(trigdat.trigtime, format='fermi')
        frame = trigdat.poshist.at(tig_time)
    except Exception as e:
        print(f"Failed to get trigdat for {trigger_number}: {e}")     

    try:        
        detectors_data = {det.name: frame.detector_angle(det.name, source_coord).deg[0] for det in GbmDetectors}
        detectors_data_sorted = sorted(detectors_data.items(), key=lambda x: x[1])
        problem_detectors = check_GBM_status(tig_time)

        # Filter for NaI detectors with viewing angle <= 60 deg not in problem list
        selected = [(name, angle) for name, angle in detectors_data_sorted 
                    if angle <= 60.0 and name[0] != 'b' and name not in problem_detectors]
        
        det_list = [GbmDetectors.from_full_name(GbmDetectors.from_str(name).full_name).name for name, _ in selected]   
        
        if not det_list:
            nai_det_list = [(name, angle) for name, angle in detectors_data_sorted if name[0] != 'b' and name not in problem_detectors]
            det_list.append(nai_det_list[0][0])
        return det_list
    except Exception as e:
        print(f"Failed to get detectors list for {trigger_number}: {e}")
        return 'all'
    
def normalize_det_list(det_list):
    """Ensures detector list is consistently formatted."""
    if isinstance(det_list, str):
        return [d.strip() for d in det_list.split(",") if d.strip()]
    elif isinstance(det_list, list):
        return det_list
    return []

def config_to_det_list(config_dic, trigger_directory):
    """Parses detector requirements from the config file."""
    det_list = config_dic.get("det_list", None)
    if not det_list or det_list == 'None' or det_list == 0:
        return None, None
        
    if config_dic["det_list"] in ['best', 'one']:
        nai_dets = get_dets_list(config_dic['trigger_number'], trigger_directory)
        if config_dic["det_list"] == 'one': nai_dets = [nai_dets[0]]
        det_string = "_".join(nai_dets)
    elif config_dic["det_list"] != "all":
        config_dic["det_list"] = normalize_det_list(config_dic["det_list"])
        nai_dets = [d for d in config_dic['det_list'] if d.startswith('n')]
        det_string = "_".join(nai_dets)
    else:
        nai_dets = [f'n{i}' for i in range(10)] + ['na', 'nb']
        det_string = "all"
    return nai_dets, det_string

def sanitize_for_yaml(obj):
    """Recursively converts NumPy types to native Python types for clean YAML output."""
    if isinstance(obj, np.ndarray):
        return [sanitize_for_yaml(i) for i in obj.tolist()]
    elif isinstance(obj, np.generic):
        return obj.item()  # Converts np.float64 to float, np.int64 to int, etc.
    elif isinstance(obj, dict):
        return {k: sanitize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_yaml(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_yaml(i) for i in obj)
    return obj

# ==========================================
# 3. Core Signal & Metric Analysis 
# ==========================================

def calculate_snr(lc, back_intervals):
    """Calculates the signal-to-noise ratio (SNR) for a given light curve."""
    times, counts = lc.centroids, lc.counts
    bkg_mask = np.zeros_like(times, dtype=bool)
    for t_start, t_stop in back_intervals:
        bkg_mask |= (times >= t_start) & (times < t_stop)
    
    bkg_counts = counts[bkg_mask]
    if len(bkg_counts) == 0: return np.nan
        
    avg_bkg_per_bin = np.mean(bkg_counts)
    src_counts = counts[~bkg_mask]
    if len(src_counts) == 0: return 0.0
        
    peak_counts = np.max(src_counts)
    net_signal = peak_counts - avg_bkg_per_bin
    
    if avg_bkg_per_bin > 0:
        return net_signal / np.sqrt(avg_bkg_per_bin)
    return np.inf if net_signal > 0 else 0.0

def define_time_intervals(trigger_number, trigger_directory):
    """Defines optimal source and background time intervals dynamically based on T90."""
    try:
        print("Fetching GRB parameters for time windowing...")
        T0, T90, T50, PF64, PFLX, FLU = get_GRB_par(trigger_number, trigger_directory)
        source_par_dict = {"T0": T0, "T90": T90, "T50": T50, "PF64": PF64, "PFLX": PFLX, "FLU": FLU}
        
        if T90 < 2.0:
            padding = 0.256 * 10 + T90
            back_before = [T0 - padding - 50, T0 - padding]
            back_after = [T0 + T90 + padding * 2, T0 + T90 + padding * 2 + 50]
        elif 2.0 <= T90 < 6.0:
            padding = 1.024 * 10 + T90 * 0.2
            reference = min(110, max(50.0, T90)) + padding
            back_before = [T0 - padding - 50, T0 - padding]
            back_after = [T0 + T90 + padding, T0 + T90 + reference]
        else:
            padding = 1.024 * 10 + T90 * 0.2
            r_padding = 1.024 * 10 + T90 * 0.4
            reference = min(110, max(50.0, T90)) + padding
            r_reference = min(110, max(50.0, T90)) + r_padding
            back_before = [T0 - reference, T0 - padding]
            back_after = [T0 + T90 + r_padding, T0 + T90 + r_reference]
        
        src_range = (T0 - padding / 2, T0 + T90 + padding * 0.9)
        back_intervals = [back_before, back_after]
    except Exception as e:
        print(f"Failed to get GRB params: {e}. Using default window.")
        src_range, back_intervals = (-10, 120), [[-50, -10], [120, 150]]
        source_par_dict = None

    trange_total = (back_intervals[0][0], back_intervals[1][1])
    print(f"Source window: {src_range[0]:.2f} - {src_range[1]:.2f}")
    print(f"Background intervals: {back_intervals}")
    return src_range, back_intervals, trange_total, source_par_dict

def generate_lightcurves(tte_files, src_range, trange_total, bw=0.064, erange=(8.0, 900.0), combined=False, src_only=False):
    """Generates individual and/or combined light curves from TTE files."""
    lcs_src, lcs_total, titles, tte_list = [], [], [], []
    for tte_file in tte_files:
        tte = GbmTte.open(tte_file).slice_time(src_range if src_only else trange_total)
        tte_list.append(tte)

        if not combined:
            phaii = tte.to_phaii(bin_by_time, bw)
            lcs_src.append(phaii.to_lightcurve(time_range=src_range, energy_range=erange))
            lcs_total.append(phaii.to_lightcurve(energy_range=erange))
            titles.append(os.path.basename(tte_file))
        
    lc_combined_src, lc_combined_total = None, None
    if tte_list:
        tte_combined = GbmTte.merge(tte_list)
        phaii_combined = tte_combined.to_phaii(bin_by_time, bw)
        lc_combined_src = phaii_combined.to_lightcurve(time_range=src_range, energy_range=erange)
        if not src_only:
            lc_combined_total = phaii_combined.to_lightcurve(energy_range=erange)

    return lcs_src, lcs_total, lc_combined_src, lc_combined_total, titles

def find_optimal_detectors(trigger_number, trigger_directory, bw=0.064, config_dict=None, plot_flag=False):
    """Analyzes all detectors to find the combination that maximizes the combined SNR."""
    if config_dict is None:
        src_range, back_intervals, trange_total, source_par_dict = define_time_intervals(trigger_number, trigger_directory)
    else:
        src_range, back_intervals, trange_total = config_dict["src_range"], config_dict["back_intervals"], config_dict["trange_total"]
        source_par_dict = None

    all_tte_files, _ = trigger_file_list(trigger_directory, "tte", trigger_number)

    print("\n--- Ranking individual detectors ---")
    detector_snrs = []
    for tte_file in all_tte_files:
        _, lc_total, _, _, _ = generate_lightcurves([tte_file], src_range, trange_total, bw)
        snr = calculate_snr(lc_total[0], back_intervals)
        name = os.path.basename(tte_file)
        print(f"  {name}: SNR = {snr:.2f}")
        detector_snrs.append({'file': tte_file, 'name': name, 'snr': snr})
    
    ranked_detectors = sorted(detector_snrs, key=lambda x: x['snr'], reverse=True)

    print("\n--- Finding optimal combination ---")
    snr_evolution = []
    for k in range(1, len(ranked_detectors) + 1):
        files_to_combine = [d['file'] for d in ranked_detectors[:k]]
        _, _, _, lc_combined_total, _ = generate_lightcurves(files_to_combine, src_range, trange_total, bw, combined=True)
        combined_snr = calculate_snr(lc_combined_total, back_intervals)
        snr_evolution.append({'k': k, 'snr': combined_snr})
        print(f"  Combined {k} detectors... New SNR = {combined_snr:.2f}")

    best_result = max(snr_evolution, key=lambda x: x['snr'])
    best_k = best_result['k']
    optimal_files = [d['file'] for d in ranked_detectors[:best_k]]
    
    print(f"\nMaximum SNR of {best_result['snr']:.2f} found with the top {best_k} detectors.")
    
    if plot_flag:
        return optimal_files, snr_evolution, ranked_detectors, src_range, back_intervals, trange_total, source_par_dict
    return optimal_files, snr_evolution, src_range, back_intervals, trange_total, source_par_dict

# ==========================================
# 4. Plotting Generation
# ==========================================

def plot_gbm_lightcurves(trigger_directory, tte_files, src_range, back_intervals, bw=0.064, suffix=""):
    """Generates and saves high-quality PDF grid plots of GBM light curves."""
    setup_publication_style()
    trange_total = (back_intervals[0][0], back_intervals[1][1])
    lcs_src, lcs_total, lc_comb_src, lc_comb_total, titles = generate_lightcurves(tte_files, src_range, trange_total, bw)

    num_plots = len(lcs_src) + (1 if lc_comb_src else 0)
    cols, rows = 4, int(np.ceil(num_plots / 4))

    def _draw_plots(axes, is_shared_y):
        ax_flat = axes.flatten()
        for i, (lc_s, lc_t, title) in enumerate(zip(lcs_src, lcs_total, titles)):
            ax = ax_flat[i]
            det_name = title.split('_')[2]
            ax.step(lc_s.centroids, lc_s.counts, where='post')
            ax.grid(True, linestyle='--', alpha=0.5)
            snr = calculate_snr(lc_t, back_intervals)
            ax.text(0.95, 0.95, f"Det: {det_name.lower()}\nSNR: {snr:.1f}", transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8), fontsize=12)

        if lc_comb_src:
            ax = ax_flat[len(lcs_src)]
            ax.step(lc_comb_src.centroids, lc_comb_src.counts, where='post', color='black', linewidth=2.0)
            ax.set_title('Combined')
            ax.grid(True, linestyle='--', alpha=0.5)
            snr = calculate_snr(lc_comb_total, back_intervals)
            ax.text(0.95, 0.95, f"SNR: {snr:.1f}", transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8), fontsize=12)

        for j in range(num_plots, len(ax_flat)): ax_flat[j].axis('off')

    # Plot 1: Shared Y-Axes
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True, sharey=True)
    _draw_plots(axes1, is_shared_y=True)
    fig1.supxlabel('Time since trigger (s)')
    fig1.supylabel('Counts / bin')
    fig1.suptitle(f'Detector Light Curves (Shared Y, {bw * 1000:.1f} ms){suffix}')
    
    output_path1 = Path(trigger_directory) / f"lc_grid_shared_y{suffix}_{bw * 1000:.0f}ms.pdf"
    fig1.savefig(output_path1); plt.close(fig1)
    print(f"✅ Saved shared-axis plot: \n   {output_path1}")

    # Plot 2: Independent Y-Axes
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True, sharey=False)
    _draw_plots(axes2, is_shared_y=False)
    fig2.supxlabel('Time since trigger (s)')
    fig2.supylabel('Counts / bin')
    fig2.suptitle(f'Detector Light Curves (Independent Y, {bw * 1000:.1f} ms){suffix}')
    
    output_path2 = Path(trigger_directory) / f"lc_grid_indep_y{suffix}_{bw * 1000:.0f}ms.pdf"
    fig2.savefig(output_path2); plt.close(fig2)
    print(f"✅ Saved independent-axis plot: \n   {output_path2}")

def full_analysis_workflow(trigger_num, trigger_dir, bin_width=0.064, config_dict=None):
    """Executes full analysis finding optimal detectors and producing evolution plots."""
    trigger_dir = Path(trigger_dir)
    best_detector_files, snr_results, ranked_detectors, src_range, back_intervals, trange_total, source_par_dict = find_optimal_detectors(
        trigger_num, trigger_dir, bw=bin_width, config_dict=config_dict, plot_flag=True
    )
    
    print("\n--- Final Result ---")
    print(f"Optimal detectors: {[os.path.basename(f)[8:10] for f in best_detector_files]}\n")

    plot_gbm_lightcurves(trigger_dir, best_detector_files, src_range, back_intervals, bw=bin_width, suffix="_optimal")

    # --- Enhanced SNR Evolution Plot ---
    setup_publication_style()
    k_values = [res['k'] for res in snr_results]
    combined_snr_values = [res['snr'] for res in snr_results]
    individual_snr_values = [d['snr'] for d in ranked_detectors]
    detector_names = [os.path.basename(d['file']).split('_')[2].upper() for d in ranked_detectors]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(k_values, individual_snr_values, color='gray', alpha=0.5, label='Individual Detector SNR')
    ax.plot(k_values, combined_snr_values, 'o-', color='royalblue', linewidth=2.5, markersize=8, label='Cumulative Combined SNR')

    ax.set_xlabel('Detectors Added in Order of Rank')
    ax.set_ylabel('Signal-to-Noise Ratio (SNR)')
    ax.set_xticks(k_values)
    ax.set_xticklabels(detector_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.5, axis='y')

    best_k_result = max(snr_results, key=lambda x: x['snr'])
    best_k, best_snr = best_k_result['k'], best_k_result['snr']
    
    ax.axvline(best_k, color='crimson', linestyle='--', label=f'Optimal k={best_k} (SNR={best_snr:.2f})')
    ax.plot(best_k, best_snr, 'o', markersize=12, color='crimson', fillstyle='none', markeredgewidth=2)
    ax.legend()
    
    snr_evo_path = trigger_dir / f"snr_evolution_bn{trigger_num}_{int(bin_width*1000)}ms.pdf"
    fig.savefig(snr_evo_path)
    plt.close(fig)
    print(f"✅ Enhanced SNR evolution plot saved to: \n   {snr_evo_path}")
    
    return best_detector_files, source_par_dict


# ==========================================
# 5. Background Fitting & Simulation
# ==========================================

def fit_background(phai_all, erange, src_interval, bkgd_intervals, outpath, trigger_number):
    """Interpolates background across the source interval using optimal polynomial order."""
    lc_tot = phai_all.to_lightcurve(energy_range=erange)
    src_lc = phai_all.to_lightcurve(time_range=src_interval, energy_range=erange)
    bkg_lc_plot1 = phai_all.to_lightcurve(time_range=bkgd_intervals[0], energy_range=erange)
    bkg_lc_plot2 = phai_all.to_lightcurve(time_range=bkgd_intervals[1], energy_range=erange)

    backfitter = BackgroundFitter.from_phaii(phai_all, Polynomial, time_ranges=bkgd_intervals)
    backfitter.fit(order=1)
    best_order = find_best_poly_order(backfitter, energy_range=None, det='n1', max_order=4, outpath=outpath)
    backfitter.fit(order=best_order)

    bkgd_fit = backfitter.interpolate_bins(phai_all.data.tstart, phai_all.data.tstop)
    bkgd_fit_lc = bkgd_fit.integrate_energy(*erange)
    
    lcplot = Lightcurve(data=lc_tot, background=bkgd_fit_lc)
    lcplot.add_selection(src_lc)
    lcplot.add_selection(bkg_lc_plot1)
    lcplot.add_selection(bkg_lc_plot2)
    lcplot.selections[0].color = 'green'
    lcplot.selections[1].color = 'pink'
    lcplot.selections[2].color = 'pink'
    
    fig_name = outpath + f"selection_bn{trigger_number}.png"
    plt.savefig(fig_name); plt.close()
    print(f"Selection plot saved to {fig_name}")
    return backfitter

def run_single_simulation(sim_index, base_counts, bin_width_s, haar_python_path, time_resolved, t_start=0.0, window_size_s=None, step_size_s=None):
    """Executes a discrete Minimum Variability Timescale (MVT) simulation."""
    try:
        np.random.seed(sim_index + 1)
        counts = np.random.poisson(base_counts)

        if not time_resolved:
            mvt_res = run_mvt_in_subprocess(counts, bin_width_s=bin_width_s, haar_python_path=haar_python_path)
        else:
            mvt_res = run_mvt_in_subprocess(
                counts=counts, bin_width_s=bin_width_s, haar_python_path=haar_python_path,
                time_resolved=True, window_size_s=window_size_s, step_size_s=step_size_s, tstart=t_start
            )
        plt.close('all')
        return mvt_res
    except Exception as e:
        print(f"Error in simulation {sim_index} for bin width {bin_width_s*1000} ms: {e}")
        return None 

def create_default_config(trigger_number):
    """Generates an initial simulation configuration dict when running via trigger ID."""
    print(f"No config file provided. Generating a default configuration for bn{trigger_number}.")
    return {
        'instrument': 'GBM (default)',
        'trigger_number': trigger_number,
        'det_list': None, 
        'time_resolved': False,
        'total_sim': 30,
        'bin_width_ms': 1.0
    }

# ==========================================
# 6. Main Execution Pipeline
# ==========================================

class CustomLightcurve:
    """Mock object that tricks existing GDT functions into accepting external data."""
    def __init__(self, times, counts):
        self.centroids = np.array(times, dtype=float)
        self.counts = np.array(counts, dtype=float)


def rebin_custom_lightcurve(times, counts, orig_bw, new_bw):
    """
    Rebins 1D lightcurve arrays to a new bin width (count-preserving).
    """
    # Calculate original bin edges
    t_start = times[0] - orig_bw / 2.0
    t_stop = times[-1] + orig_bw / 2.0
    orig_edges = np.linspace(t_start, t_stop, len(counts) + 1)
    
    # Cumulative counts
    cum_counts = np.insert(np.cumsum(counts), 0, 0.0)
    
    # Define new bin edges
    new_edges = np.arange(t_start, t_stop, new_bw)
    if new_edges[-1] < t_stop:
        new_edges = np.append(new_edges, t_stop)
        
    # Interpolate cumulative counts to new edges and difference them
    cum_interp = np.interp(new_edges, orig_edges, cum_counts)
    new_counts = np.diff(cum_interp)
    new_times = new_edges[:-1] + np.diff(new_edges) / 2.0
    
    return new_times, new_counts


def main(config_dic, config_flag=False, data_bw=None, lc_times=None, lc_counts=None):
    """Drives the entire trigger processing and simulation workflow."""
    with open(MVT_CONFIG_FILE, 'r') as f:
        MVT_config = yaml.safe_load(f)

    haar_python_path = MVT_config['project_settings']['haar_python_path']
    time_resolved = config_dic.get('time_resolved', False)
    total_sim = config_dic.get('total_sim', 100)
    trigger_number = config_dic.get('trigger_number', 'custom')
    outpath = config_dic.get('output_path', Path.cwd())
    
    output_path = Path(outpath) / f"MVT_bn{trigger_number}/"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path}")

    # ==========================================
    # FORK 1: External Data Bypass
    # ==========================================
    is_custom_data = (data_bw is not None and lc_times is not None and lc_counts is not None)
    
    if is_custom_data:
        print(f"External data detected (BW={data_bw}s). Bypassing Fermi GBM download pipeline.")
        
        # Override instrument if it defaults to GBM
        if config_dic.get('instrument', 'GBM (default)') == 'GBM (default)':
            config_dic['instrument'] = 'unknown'

        bin_width_s = float(data_bw)
        bin_width_ms = bin_width_s * 1000.0
        
        # Initialize our mock object
        lc_target = CustomLightcurve(lc_times, lc_counts)
        base_counts = lc_target.counts
        
        # Extract timing directly from the user's time array
        t_start = lc_target.centroids[0] - (bin_width_s / 2)
        t_stop = lc_target.centroids[-1] + (bin_width_s / 2)
        src_interval = [t_start, t_stop]
        
        # Structural defaults for downstream configuration
        det_list = ['custom']
        det_string = "custom"
        selection_str = f"custom_data_{round(t_start, 2)}s"
        back_intervals = config_dic.get('background_intervals', [])
        
        en_lo = config_dic.get('en_lo', 0)
        en_hi = config_dic.get('en_hi', 0)
        T0 = config_dic.get('T0', 0)
        T90 = config_dic.get('T90', 0)
        trigger_directory = output_path
        mvt_time_window = config_dic.get('mvt_time_window', 0.5)
        mvt_step_size = config_dic.get('mvt_step_size', 0.5)

    # ==========================================
    # FORK 2: Standard Fermi GBM Pipeline
    # ==========================================
    else:
        trigger_number = config_dic['trigger_number']
        en_lo = config_dic.get('en_lo', 8)
        en_hi = config_dic.get('en_hi', 900)
        data_path = config_dic.get('data_path', Path.cwd())
        T0 = config_dic.get('T0', 0)
        t_start = config_dic.get('tstart', 0)
        t_stop = config_dic.get('tstop', 0)
        T90 = config_dic.get('T90', 0)
        back_intervals = config_dic.get('background_intervals', 0)
        bin_width_ms = config_dic.get('bin_width_ms', 0.1)
        mvt_time_window = config_dic.get('mvt_time_window', 0.5)
        mvt_step_size = config_dic.get('mvt_step_size', 0.5)

        print(f"Trigger number: {trigger_number}")
        trigger_directory = Path(data_path) / f"bn{trigger_number}"
        det_list, _ = config_to_det_list(config_dic, trigger_directory)
        energy_range_nai = (en_lo, en_hi)
        
        if not trigger_directory.exists():
            print(f"Trigger directory {trigger_directory} does not exist. Downloading...")
            download_data(trigger_number, trigger_directory, flags={'tte':True, 'rsp':True, 'cat':True, 'trigdat':True})

        # Determine Optimal Detectors
        if not det_list:
            print("No valid detectors found. Using 'best' based on SNR.")
            tte_list_path, source_par_dict = full_analysis_workflow(trigger_number, trigger_directory, bin_width=0.064)
            det_list = [os.path.basename(f)[8:10] for f in tte_list_path]
        else:
            tte_list_path, _ = trigger_file_list(trigger_directory, "tte", trigger_number, det_list)
        
        print(f"\n--------- Starting MVT analysis for Trigger {trigger_number} ---------")
        
        # Establish Timing Windows
        if not T90 or T90 <= 0:
            try:
                print("Fetching GRB parameters for T90...")
                T0, T90, _, _, _, _ = get_GRB_par(trigger_number, trigger_directory)
            except Exception:
                print("Failed to get T90 from BCAT. Exiting."); exit(1)

        if t_start == t_stop == 0:
            t_start = T0 - 0.5 * min(5, T90) - 1
            t_stop = T0 + T90 * 1.2 + 2

        src_interval = [t_start, t_stop]
        det_string = "".join(det_list) if det_list else "not_specified"
        selection_str = f"{round(t_start, 2)}_{round(t_stop, 2)}s_{det_string}"

        # Setup Background Intervals
        if back_intervals in [0, [[0,0],[0,0]], None, 'none', 'None', 'no']:
            print("Using default background intervals based on T90.")
            if T90 < 2.0:
                padding = 0.256 * 10 + T90
                back_before = [t_start - padding - 50, t_start - padding]
                back_after = [t_stop + padding * 2, t_stop + padding * 2 + 50]
            elif 2.0 <= T90 < 6.0:
                padding = 1.024 * 10 + T90 * 0.2
                reference = min(110, max(50.0, T90)) + padding
                back_before = [t_start - padding - 50, t_start - padding]
                back_after = [t_stop + padding, t_stop + reference]
            else:
                padding = 1.024 * 10 + T90 * 0.2
                r_padding = 1.024 * 10 + T90 * 0.4
                reference = min(110, max(50.0, T90)) + padding
                r_reference = min(110, max(50.0, T90)) + r_padding
                back_before = [t_start - reference, t_start - padding]
                back_after = [t_stop + r_padding, t_stop + r_reference]
            back_intervals = np.around(np.array([back_before, back_after]), 4).tolist()

        trange = [back_intervals[0][0]-10, back_intervals[1][1]+10]
        bin_width_s = bin_width_ms / 1000.0

        _, _, lc_target, _, _ = generate_lightcurves(tte_list_path, src_interval, trange, bin_width_s, energy_range_nai, combined=True, src_only=True)
        base_counts = lc_target.counts


    # ==========================================
    # SHARED: Run Parallel Subprocesses for MVT
    # ==========================================
    try:
        if time_resolved:
            print("\n@@@@@@ Running in time-resolved mode. @@@@@@")
            print(f"\t ⚠️ ⚠️ WARNING: Time-resolved MVT occasionally gives artificial results due to edge effects.⚠️⚠️ \n \t ⚠️⚠️ Please interpret the results with caution.⚠️⚠️")
            print(f"Time-resolved mode: window size = {mvt_time_window}s, step size = {mvt_step_size}s")

        else:
            print("\n@@@@@@ Running in standard mode. @@@@@@")
            
        task_function = functools.partial(
            run_single_simulation, base_counts=base_counts, bin_width_s=bin_width_s,
            haar_python_path=haar_python_path, time_resolved=time_resolved,
            window_size_s=mvt_time_window if time_resolved else None,
            step_size_s=mvt_step_size if time_resolved else None, t_start=t_start
        )

        print(f"\n----- Starting {total_sim} parallel simulations bin width: {bin_width_ms}ms -----")
        MVT_time_resolved_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iterator = executor.map(task_function, range(total_sim))
            MVT_time_resolved_results = [res for res in tqdm(results_iterator, total=total_sim, unit="sim") if res is not None]
    except Exception as e:
        print(f"Error during MVT calculation: {e}")
        MVT_time_resolved_results = []

    # ==========================================
    # FINAL: Summaries, Plots, and Saving
    # ==========================================
    output_info = {'trigger_number': trigger_number, 'file_path': output_path, 'selection_str': selection_str}
    
    if time_resolved:
        mvt_summary_df = analysis_mvt_time_resolved_results_to_dataframe(MVT_time_resolved_results, output_info, bin_width_ms, total_sim)
        
        # Use finer binning (1ms) for Fermi plots. 
        # For custom data, we can mathematically rebin it to 1ms for a smooth visual background.
        if not is_custom_data:
            _, _, lc_plot, _, _ = generate_lightcurves(tte_list_path, src_interval, trange, 0.064, energy_range_nai, combined=True, src_only=True)
        else:
            # Notice we removed the mvt_s reference here.
            # We just pass the 1ms rebinned data directly to the plotter!
            lc_plot = rebin_custom_lightcurve(
                lc_target.centroids, lc_target.counts, bin_width_s, 0.064
            )

        create_final_GBM_plot_with_MVT(
            lc_tot=lc_plot, 
            t_start=t_start, 
            t_stop=t_stop, 
            output_info=output_info, 
            bin_width_ms=bin_width_ms, 
            mvt_window_size_s=mvt_time_window, 
            mvt_summary_df=mvt_summary_df, 
            instrument=config_dic.get('instrument', 'unknown')
        )
    else:
        mvt_res = run_mvt_in_subprocess(
                base_counts, bin_width_s=bin_width_s, haar_python_path=haar_python_path, doplot=1,
                file_name=str(output_path) + f"/mvt_bn{trigger_number}.png"
        )
        mvt_summary, _ = analysis_mvt_results_to_dataframe(MVT_time_resolved_results, output_info, bin_width_ms, total_sim)
        
        # SNR Calculation requires TTE files, so we bypass it for external data
        if is_custom_data:
            mvt_s = mvt_summary['median_mvt_ms'] / 1000.0
            
            # Check if valid background intervals were provided
            invalid_bkg = [0, [[0,0],[0,0]], None, 'none', 'None', 'no', []]
            if back_intervals not in invalid_bkg:
                print(f"\n⚠️ WARNING: Computing tentative SNR on MVT timescale ({mvt_s:.4f}s).")
                print("   This is based on mathematical rebinning of the custom 1D lightcurve.")
                
                # Rebin the arrays to the MVT timescale
                rebinned_times, rebinned_counts = rebin_custom_lightcurve(
                    lc_target.centroids, lc_target.counts, bin_width_s, mvt_s
                )
                
                # Wrap it in our mock object and pass to your existing calculate_snr!
                rebinned_lc = CustomLightcurve(rebinned_times, rebinned_counts)
                SNR_mvt = round(calculate_snr(rebinned_lc, back_intervals), 2)
            else:
                SNR_mvt = None

        else:
            mvt_s = mvt_summary['median_mvt_ms'] / 1000.0
            _, _, _, lc_snr, _ = generate_lightcurves(tte_list_path, src_interval, trange, mvt_s, energy_range_nai, combined=True)
            SNR_mvt = round(calculate_snr(lc_snr, back_intervals), 2)
            
        mvt_summary_all = {**mvt_res, 'SNR_mvt': SNR_mvt, **mvt_summary}
        
        write_yaml(mvt_summary_all, output_path / f"mvt_summary_bn{trigger_number}_{selection_str}_{(bin_width_ms)}ms.yaml")

    # Update state and construct final YAML Configuration
    config_dic.update({
        'tstart': t_start, 'tstop': t_stop, 'background_intervals': back_intervals,
        'det_list': det_list, 'T90': T90, 'T0': T0, 'en_lo': en_lo, 'en_hi': en_hi,
        'total_sim': total_sim, 'time_resolved': time_resolved, 'bin_width_ms': bin_width_ms,
        'mvt_time_window': mvt_time_window, 'mvt_step_size': mvt_step_size,
        'det_string': det_string, 'source_interval': src_interval, 'energy_range': [en_lo, en_hi],
        'mvt_summary': mvt_summary_all if not time_resolved else mvt_summary_df.to_dict()
    })

    # --- Sanitize the dictionary to remove all NumPy types ---
    config_dic = sanitize_for_yaml(config_dic)

    if config_flag:
        final_config_path = output_path / f"config_MVT_{trigger_number}_{det_string}_sim{total_sim}_{(bin_width_ms)}ms.yaml"
    else:
        config_dic['email_flag'] = False
        final_config_path = trigger_directory / f"config_MVT_{trigger_number}.yaml"
        
    write_yaml(config_dic, final_config_path)
    print(f"Final Result & configuration saved to: \n {final_config_path}")
    
    if not time_resolved:
        print("\n" + " Results ".center(50, "@"))
        print_nested_dict(config_dic['mvt_summary'])
    print("-"*50 + "\n")

    if not time_resolved:
        # --- NEW SUBPROCESS CALL ---
        mvt_val = config_dic['mvt_summary'].get('median_mvt_ms')
        snr_val = config_dic['mvt_summary'].get('SNR_mvt')

        if mvt_val is not None and snr_val is not None:
            print(f"Running MVT Classification for MVT={mvt_val}, SNR={snr_val}...")
            
            # Call the script exactly as you would from the terminal
            cmd = [
                sys.executable, "clasify_mvt_point.py", 
                "--mvt", str(mvt_val), 
                "--snr", str(snr_val), 
                "--mode", "plot"
            ]
            
            # Run the command
            subprocess.run(cmd, check=False)
            
            # Move the generated plot from the current directory to the trigger directory
            plot_filename = f"classification_MVT_{mvt_val}_SNR_{snr_val}.png"
            if os.path.exists(plot_filename):
                destination = output_path / f"classification_MVT_{trigger_number}_{det_string}_sim{total_sim}_{bin_width_ms}ms_{mvt_val}_SNR_{snr_val}.png"
                shutil.move(plot_filename, destination)
                print(f"Classification plot moved to: \n {destination}")
        # ---------------------------

    return final_config_path

# ==========================================
# CLI Execution Handler
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MVT analysis for a Fermi GBM trigger.")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument('-c', '--config', type=str, help="Path to the simulation YAML configuration file.")
    group.add_argument('-bn', '--trigger_number', type=str, help="Trigger number (e.g., '080916009'). Default config will be generated.")
    args = parser.parse_args()
    
    email_flag = False
    config = None
    
    if args.config:
        print(f"Loading configuration from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            email_flag = config.get('email_flag', False) 
        config_flag = True
    elif args.trigger_number:
        config = create_default_config(args.trigger_number)
        config_flag = False

    if config:
        mvt_result_path = main(config, config_flag)
        if email_flag:
            send_email(
                subject=f"Button!! Analysis Complete for {config['trigger_number']}:",
                body=f"The MVT analysis for trigger {config['trigger_number']} is complete.",
                attachment_path=mvt_result_path
            )
    else:
        print("Error: Could not create or load a configuration. Exiting.")