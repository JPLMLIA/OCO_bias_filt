# _local_hardcoded_paths.py
# This file contains the original hardcoded paths for your local testing.
# It is NOT part of the open-source repository and should be in .gitignore.
# It is structured to be a drop-in replacement for paths.py for local execution.

import os
from pathlib import Path
from typing import Optional

# --- Project Root (this script's directory) ---
# This assumes this file, when renamed to paths.py, is in the root of your project 'code/'
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Input Data Directories (Hardcoded Absolute Paths) ---
# These are derived from your original _local_hardcoded_paths.py file
BASE_DATA_DIR = Path("/Volumes/OCO") # Common base from your paths

OCO_LITE_FILES_DIR = BASE_DATA_DIR / "LiteFiles" / "B11.2_OCO2"
TCCON_FILES_DIR = BASE_DATA_DIR / "TCCON/ggg2020"
MODEL_INPUT_DATA_DIR = BASE_DATA_DIR / "Models" # Assuming 'Models' dir under /Volumes/OCO
CLOUD_DATA_DIR = BASE_DATA_DIR / "3D_cloud_metrics_OCO2_V9"
EMISSION_DATA_DIR = BASE_DATA_DIR / "TOTALS_emi_nc"

# --- Output & Working Directories (Hardcoded Absolute Paths or Relative to Project Root) ---

# Main export directory (can be project-relative or absolute if preferred)
EXPORT_DIR = BASE_DATA_DIR / "LiteFiles" / "export" 

# Subdirectories for outputs (using absolute paths derived from original file)
PAR_DIR = BASE_DATA_DIR / "Parquet_OCO2_B112"
PRELOAD_DIR = PAR_DIR # Preloaded data will be stored in the same directory as Parquet files

# Directory for saving trained bias correction models (project-relative)
MODEL_SAVE_DIR = PROJECT_ROOT / "bias_corr_models"

# Directory for filter models (project-relative)
FILTER_DIR = PROJECT_ROOT / "filter_models" / "current_filters"

# Directory for figures/plots
# Using the specific path from your original file for feature selection plots
# and assuming it can serve as the general FIGURE_DIR
FIGURE_DIR = Path("/Users/smauceri/Projects/OCO2/OCO_Bias_Filt/plots/")
# If you want it project-relative like in paths.py, it would be:
# FIGURE_DIR = PROJECT_ROOT / "visualization_scripts" / "figures"

# --- Pipeline Script Directories (relative to PROJECT_ROOT) ---
BIAS_CORRECTION_DIR = PROJECT_ROOT / "bias_correction"
VISUALIZATION_SCRIPTS_DIR = PROJECT_ROOT / "visualization_scripts"

# --- Utility file (relative to PROJECT_ROOT) ---
UTIL_FILE = PROJECT_ROOT / "util.py"

# --- Specific Model Paths (B11.2 example) ---
# These mirror the structure in paths.py but use the MODEL_SAVE_DIR and FILTER_DIR defined *in this file*.
# Bias Correction Models
TC_LND_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_xco2_TCCON_biasLndNDGL_lnd_RF0'
TC_OCN_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_xco2_TCCON_biasSeaGL_sea_RF0'
SA_LND_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_prec_xco2raw_SA_biasLndNDGL_lnd_RF0'
SA_OCN_CORR_MODEL = MODEL_SAVE_DIR / 'B11.2/V1_11.2_2.6_prec_xco2raw_SA_biasSeaGL_sea_RF0'

# Filter Models
TC_LND_FILTER_MODEL = FILTER_DIR / 'tc_lnd_rev.joblib'
TC_OCN_FILTER_MODEL = FILTER_DIR / 'tc_ocn_rev.joblib'
SA_LND_FILTER_MODEL = FILTER_DIR / 'sa_lnd_rev.joblib'
SA_OCN_FILTER_MODEL = FILTER_DIR / 'sa_ocn_rev.joblib'

# --- Function to construct specific preload file paths ---
def get_preload_filepath(mode: str, qf: Optional[int], year: int) -> Path:
    """
    Constructs the full path to a preloaded data file.

    Args:
        mode (str): The mode (e.g., 'LndNDGL', 'SeaGL').
        qf (int | None): The quality flag. Appears as 'None' in filename if None.
        year (int): The year of the data.

    Returns:
        Path: The full Path object to the preloaded .parquet file.
    """
    
    qf_str = str(qf) if qf is not None else "None"
    filename = f'PreLoadB112v2_balanced_5M_{mode}_qf{qf_str}_{year}.parquet'
    return PRELOAD_DIR / filename

# --- Function to ensure directories exist ---
def ensure_dir_exists(path_to_check: Path):
    """Checks if a directory exists, and creates it if it doesn't."""
    if not path_to_check.exists():
        print(f"Directory {path_to_check} does not exist. Creating it.")
        path_to_check.mkdir(parents=True, exist_ok=True)
    elif not path_to_check.is_dir():
        raise NotADirectoryError(f"{path_to_check} exists but is not a directory.")

if __name__ == "__main__":
    print("--- Defined Paths (from _local_hardcoded_paths.py) ---")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Using BASE_DATA_DIR: {BASE_DATA_DIR}")

    print(f"\nInput data locations (hardcoded):")
    print(f"  OCO Lite Files Dir: {OCO_LITE_FILES_DIR}")
    print(f"  TCCON Files Dir: {TCCON_FILES_DIR}")
    print(f"  Model Input Data Dir: {MODEL_INPUT_DATA_DIR}") # For model files like .h5
    print(f"  Cloud Data Dir: {CLOUD_DATA_DIR}")
    print(f"  Emission Data Dir: {EMISSION_DATA_DIR}")

    print(f"\nOutput directories (hardcoded or project-relative):")
    print(f"  Export Directory (project-relative): {EXPORT_DIR}")
    print(f"  PAR Files Directory (hardcoded): {PAR_DIR}")
    print(f"  Preloaded Data Directory (hardcoded): {PRELOAD_DIR}")
    print(f"  Trained Models Directory (project-relative): {MODEL_SAVE_DIR}")
    print(f"  Output Figures Directory (project-relative): {FIGURE_DIR}")
    print(f"  Filter Models Directory (project-relative): {FILTER_DIR}")

    print(f"\nSpecific B11.2 Models (project-relative):")
    print(f"  TC LND CORR: {TC_LND_CORR_MODEL}")
    print(f"  TC OCN CORR: {TC_OCN_CORR_MODEL}")
    print(f"  SA LND CORR: {SA_LND_CORR_MODEL}")
    print(f"  SA OCN CORR: {SA_OCN_CORR_MODEL}")
    print(f"  TC LND FILTER: {TC_LND_FILTER_MODEL}")
    print(f"  TC OCN FILTER: {TC_OCN_FILTER_MODEL}")
    print(f"  SA LND FILTER: {SA_LND_FILTER_MODEL}")
    print(f"  SA OCN FILTER: {SA_OCN_FILTER_MODEL}")

    print(f"\nScript directories (project-relative):")
    print(f"  Bias Correction Scripts: {BIAS_CORRECTION_DIR}")
    print(f"  Visualization Scripts: {VISUALIZATION_SCRIPTS_DIR}")
    print(f"  Utility File: {UTIL_FILE}")

    print("\n--- Example: Ensuring output directories exist ---")
    ensure_dir_exists(PAR_DIR)
    ensure_dir_exists(PRELOAD_DIR)
    ensure_dir_exists(FIGURE_DIR)
    ensure_dir_exists(MODEL_SAVE_DIR)
    ensure_dir_exists(FILTER_DIR)
    print("Checked/created relevant output directories.")