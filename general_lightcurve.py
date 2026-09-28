"""
general_lightcurve.py
Wrapper script to run MVT analysis on external lightcurve data.

To use custom data formats (FITS, HDF5, custom CSVs), simply 
modify the `read_lightcurve_data` function below.
"""

import argparse
import yaml
import numpy as np
from pathlib import Path

# Import the main function from the updated trigger_process pipeline
from trigger_process import main

def read_lightcurve_data(filename):
    """
    Reads the lightcurve data from a file.
    
    Users can modify this function to support their specific data formats 
    (e.g., astropy.io.fits for FITS files, pandas for complex CSVs).
    
    Args:
        filename (str or Path): Path to the data file.
        
    Returns:
        tuple: (lc_times, lc_counts, data_bw)
            - lc_times (np.ndarray): 1D array of bin center times.
            - lc_counts (np.ndarray): 1D array of counts per bin.
            - data_bw (float): Bin width in seconds.
    """
    print(f"Loading external lightcurve data from: {filename}")
    try:
        # Default implementation: reads a 2-column text file (Time, Counts)
        # np.loadtxt automatically handles whitespace and ignores '#' comments
        data = np.loadtxt(filename, comments='#')
        
        if len(data.shape) != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns, got shape {data.shape}")
            
        lc_times_start = data[:, 0]
        lc_times_stop = data[:, 1]
        lc_times = (lc_times_start + lc_times_stop) / 2
        lc_counts = data[:, 2]
        
        # Infer bin width using the median difference between time bins 
        # (median prevents floating-point jitter from affecting the result)
        data_bw = round(float(np.median(np.diff(lc_times))), 6)
        
        return lc_times, lc_counts, data_bw

    except Exception as e:
        raise RuntimeError(f"Failed to read lightcurve data from {filename}:\n{e}")



def create_default_config(file_path, lc_times, data_bw):
    """Generates a default configuration dictionary for external lightcurve data."""
    base_name = Path(file_path).stem
    t_start = float(lc_times[0] - (data_bw / 2))
    t_stop = float(lc_times[-1] + (data_bw / 2))
    
    return {
        'trigger_number': base_name,  # Uses the filename as the trigger ID by default
        'instrument': 'unknown',
        'time_resolved': False,
        'total_sim': 30,             # Default number of simulations
        'mvt_time_window': 0.5,
        'mvt_step_size': 0.1,
        'tstart': t_start,
        'tstop': t_stop,
        'output_path': str(Path.cwd()),
        'background_intervals': []    # No background subtraction by default
    }


def run_external_analysis(data_file, config_file=None):
    """Orchestrates data loading, config setup, and pipeline execution."""
    data_path = Path(data_file)
    
    if not data_path.exists():
        print(f"Error: Could not find input file '{data_file}'")
        return

    # 1. Read the data using the customizable function
    try:
        lc_times, lc_counts, data_bw = read_lightcurve_data(data_path)
        print(f"Data loaded successfully. Inferred bin width: {data_bw} seconds ({data_bw * 1000} ms)")
    except RuntimeError as e:
        print(e)
        return

    # 2. Setup Configuration
    if config_file and Path(config_file).exists():
        print(f"Loading user configuration from: {config_file}")
        with open(config_file, 'r') as f:
            config_dic = yaml.safe_load(f)
        config_flag = True
    else:
        print("No valid config provided. Generating default configuration.")
        config_dic = create_default_config(data_file, lc_times, data_bw)
        config_flag = False

    # 3. Call the Core Pipeline
    print("\n" + "="*60)
    print(f" Starting MVT pipeline for: {config_dic.get('trigger_number', 'Custom Data')}")
    print("="*60)
    
    try:
        final_config_path = main(
            config_dic=config_dic, 
            config_flag=config_flag, 
            data_bw=data_bw, 
            lc_times=lc_times, 
            lc_counts=lc_counts
        )
        print(f"\n✅ Pipeline finished successfully.")
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run MVT analysis on custom lightcurve data."
    )
    parser.add_argument(
        '-f', '--file', 
        type=str, 
        required=True, 
        help="Path to the input data file."
    )
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        help="Path to an optional YAML configuration file."
    )
    
    args = parser.parse_args()
    run_external_analysis(args.file, args.config)