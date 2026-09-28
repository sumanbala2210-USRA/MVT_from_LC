# MVT_Measurement 🚀

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-Apache_2.0-orange.svg)

A Python package for Minimum Variability Timescale (MVT) analysis, tailored for Fermi GBM data and generalized light curves.

---
## Installation

**Clone the repository:**
```bash
git clone https://github.com/sumanbala2210-USRA/MVT_from_LC.git
cd MVT_from_LC
```

## Two Python Environments Required

**IMPORTANT — Please install `conda` or `miniconda` first.** 
The HAAR environment (ENV-B) requires Python 3.10.8, which is easiest to create with conda. 
(ENV-A may be created using either conda *or* venv.)

1. Install `CONDA` or `MINICONDA`

### ENV‑A (Main Analysis Environment)

```bash
conda create -n MVT python=3.10
conda activate MVT
pip install .
```

Optional UI + development extras:

```bash
pip install .[ui,dev]
```

---

### ENV‑B (HAAR Environment)
The HAAR MVT estimator must run under specific versions to reproduce the published values.

Example HAAR environment creation:

```bash
conda create -n haar_env python=3.10.8
conda activate haar_env
pip install -r requirements_haar.txt
```

Find the python executable for ENV‑B:

```bash
which python
```

Then set that path inside `simulations_ALL.yaml`:

```yaml
project_settings:
  haar_python_path: "/path/to/conda/envs/haar_env/bin/python"
```

---

## Verify Setup

Run:

```bash
conda activate MVT
python test_mvt.py
```

This script:
1. Runs HAAR MVT inside the current Python environment.
2. Runs HAAR MVT through ENV‑B via subprocess.
3. Prints both results for comparison.

Final summary CSV files are included so values can be verified directly.

---

## Core Tools & Usage

This toolkit provides two primary workflows: one automated for **Fermi GBM** triggers, and a generalized wrapper for **any external light curve**.

### 1. Fermi GBM Analysis (`trigger_process.py`)

This script automatically downloads TTE data from AWS, calculates optimal time windows based on T90, ranks detectors by SNR to find the optimal combination, fits polynomial backgrounds, and runs parallel HAAR MVT simulations.

**Basic Run (Auto-generates configuration):**
Run the analysis by simply providing the Fermi trigger number. The script will fetch the data, determine parameters, and save a default configuration file inside a newly created data folder.
```bash
python trigger_process.py -bn 260208412
```
*(Note: This generates a `config` (YAML) file which can be used in future analysis).*

**Advanced / Iterative Run (Using the configuration):**
To run time-resolved analysis or tweak intervals, edit the generated YAML file and pass it back into the script. It will skip downloading and use your new settings:
```bash
python trigger_process.py -c bn260208412/config_MVT_260208412.yaml
```

---

## Configuration File Example

When you run a burst for the first time, a YAML configuration file is automatically generated. You can modify this file to switch to **time-resolved mode**, change the number of simulations, or define custom background intervals.

```yaml
instrument: 'GBM (default)'
trigger_number: '260208412'
det_list: [n6, n9, n7, nb, na, n2]
time_resolved: Yes                 # Change to 'Yes' or 'True' for sliding window analysis
total_sim: 30                      # Number of Poisson simulations
bin_width_ms: 1.0                  # Binning resolution
tstart: 0.5961
tstop: 180.5893
background_intervals: [[-148.7261, -38.7261], [248.9937, 358.9937]]
T90: 145.411
T0: 4.0961
en_lo: 8
en_hi: 900
mvt_time_window: 0.5               # Window size for time-resolved mode
mvt_step_size: 0.5                 # Step size for time-resolved mode
det_string: 'n6n9n7nbnan2'
source_interval: [0.5961, 180.5893]
energy_range: [8, 900]
mvt_summary:                       # Automatically populated after a successful run
    mvt_ms: 61.661
    mvt_err_ms: 7.65
    SNR_mvt: 58.14
    median_mvt_ms: 61.501
    mvt_err_lower_ms: 8.787
    mvt_err_upper_ms: 11.6046
    successful_runs: 20
    total_sim: 30
    failed_runs: 10
email_flag: False
```

---

## Pipeline Outputs

The pipeline cleanly separates raw data downloads from analytical results into two sibling folders. This prevents clutter, especially during time-resolved analysis which generates dozens of window-specific plots.

For a trigger like `260208412`, your directory will look like this:

```text
.
├── bn260208412/                              # 📥 Data & Preprocessing
│   ├── glg_tte_n0_bn260208412_v00.fit        # Downloaded TTE, CSPEC, TRIGDAT files
│   ├── lc_grid_shared_y_optimal_64ms.pdf     # Detector light curve grid plots
│   ├── snr_evolution_bn260208412_64ms.pdf    # Detector optimization curve
│   └── config_MVT_260208412.yaml             # ⚙️ Auto-generated config for re-runs
│
└── MVT_bn260208412/                          # 📊 Final Analytical Results
    ├── mvt_summary_bn...yaml                 # Final computed MVT, errors, and SNR
    │
    │   # IF STANDARD MODE:
    ├── classification_MVT_..._SNR.png        # MVT vs. SNR classification plot
    ├── mvt_bn260208412_haar_mod.png          # HAAR wavelet power spectrum plot
    │
    │   # IF TIME-RESOLVED MODE (time_resolved: True):
    ├── LC_with_MVT_...png                    # Light curve with MVT color-coded overlay
    ├── Detailed_260208412_...csv             # Full breakdown of all sliding windows
    ├── MVT_dist_..._T12.85s.png              # Distribution plot for time window T12.85s
    ├── MVT_dist_..._T14.35s.png              # Distribution plot for time window T14.35s
    └── MVT_dist_...                          # (Continues for all valid time slices)
```
---
### 2. General External Light Curves (`general_lightcurve.py`)

This script allows you to run the exact same MVT simulation pipeline on your own custom data (e.g., Swift, XMM-Newton, or simulated light curves) without needing Fermi TTE `.fit` files. 

**Input Format:** 
By default, the script expects a 3-column text file containing the bin start time, bin stop time, and counts:
```text
# t_start(s)   t_stop(s)   Counts
10.0001        10.0002     5
10.0002        10.0003     12
10.0003        10.0004     8
...
```
*(Note: If you have FITS or HDF5 files, you can easily modify the `read_lightcurve_data()` function inside `general_lightcurve.py` to parse them).*

**Basic Run:**
Provide your data file. The script will automatically calculate the bin centers, infer the bin width, bypass the Fermi-specific modules, and run the MVT simulations.
```bash
python general_lightcurve.py -f LLE_hist.txt
```
*(Note: This generates a `config` (YAML) file which can be used in future analysis).*

**Advanced / Iterative Run (Using the configuration):**
You can define custom time ranges, enable time-resolved MVT analysis, and set background intervals using a YAML configuration file.
```bash
python general_lightcurve.py -f LLE_hist.txt -c config_MVT_general.yaml
```