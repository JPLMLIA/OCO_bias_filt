# make a few plots of the exported lite files to make sure they are correct
# 10/2022 Steffen Mauceri

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent automatic display
import numpy as np
import pandas as pd
import glob
import netCDF4 as nc
from tqdm import tqdm
import os
from pathlib import Path
from util import plot_map, dist
import paths

def get_all_headers_with_dims(f):
    headers = []
    dims = []
    # Get variables in the root group
    for var_name in f.variables.keys():
        var = f.variables[var_name]
        headers.append(var_name)
        dims.append(var.ndim)
    # Get variables in subgroups
    groups = list(f.groups.keys())
    for g in groups:
        for var_name in f[g].variables.keys():
            full_var_name = g + '/' + var_name
            var = f[g].variables[var_name]
            headers.append(full_var_name)
            dims.append(var.ndim)
    return headers, dims


def load_tccon_data():
    """
    Load TCCON data from NetCDF files.
    Returns a DataFrame with TCCON measurements.
    """
    print('Loading TCCON data...')
    
    # TCCON data parameters
    time_max = 1*60*60  # 1 hour in seconds
    lat_lon_max = 2  # 2 degrees
    
    # TCCON data path
    tccon_data_path = paths.TCCON_FILES_DIR
    
    # Variables to extract from TCCON files
    t_vars = ['lat', 'long', 'xco2_x2019', 'time', 'year', 'day']
    
    # Load TCCON data
    data_all = []
    name_all = []
    TCCON_files = sorted(list(Path(tccon_data_path).glob('*.nc')))
    
    for f in TCCON_files:
        t_ds = nc.Dataset(f)
        t_ids = t_ds['lat'][:]
        # initialize array
        t_data = np.ones((len(t_ids), len(t_vars))) * np.nan
        
        i = -1
        for v in t_vars:
            i += 1
            t_data[:, i] = t_ds[v][:]
        
        # append data to list
        data_all.append(t_data)
        # add name
        name = t_ds.long_name
        # remove numbers from name
        name = name.split('0')[0]
        # add to name_all list
        name_all.append([name] * len(t_data))
    
    # merge TCCON data and names
    data_all = np.concatenate(data_all)
    name_all = np.concatenate(name_all)
    
    t_data = pd.DataFrame(data_all, columns=t_vars)
    t_data['tccon_name'] = name_all
    
    # rename xco2_x2019 to xco2
    t_data.rename(columns={'xco2_x2019': 'xco2'}, inplace=True)
    
    print(f'Loaded {len(t_data)} TCCON measurements from {len(TCCON_files)} files')
    return t_data


def match_tccon_to_soundings(data, tccon_data):
    """
    Match OCO-2 soundings to TCCON measurements.
    Returns data with added TCCON columns.
    """
    print('Matching soundings to TCCON...')
    
    time_max = 1*60*60  # 1 hour in seconds
    lat_lon_max = 2  # 2 degrees
    
    # Initialize TCCON columns
    data['xco2tccon'] = np.nan
    data['tccon_name'] = ''
    data['tccon_dist'] = np.nan
    
    # Get unique years in the data
    years = data['time'].apply(lambda x: pd.Timestamp(x, unit='s').year).unique()
    
    for year in years:
        print(f'Processing year: {year}')
        
        # Filter data for this year
        year_data = data[data['time'].apply(lambda x: pd.Timestamp(x, unit='s').year) == year]
        
        # Filter TCCON data for this year
        tccon_year = tccon_data[tccon_data['year'] == year]
        
        if len(tccon_year) == 0:
            continue
            
        # Process in daily chunks for efficiency
        for day in tqdm(range(365), desc=f'Processing {year}'):
            day_time = day * 24 * 60 * 60 + year_data['time'].iloc[0] if len(year_data) > 0 else 0
            
            # Subset data for this day
            data_day = year_data[(year_data['time'] >= day_time) & 
                                (year_data['time'] < day_time + 24 * 60 * 60)]
            tccon_day = tccon_year[(tccon_year['time'] >= day_time - time_max) & 
                                  (tccon_year['time'] < day_time + 24 * 60 * 60 + time_max)]
            
            if len(data_day) == 0 or len(tccon_day) == 0:
                continue
                
            # Get coordinates
            lat_oco = data_day['latitude'].values
            lon_oco = data_day['longitude'].values
            time_oco = data_day['time'].values
            
            lat_tccon = tccon_day['lat'].values
            lon_tccon = tccon_day['long'].values
            time_tccon = tccon_day['time'].values
            xco2_tccon = tccon_day['xco2'].values
            name_tccon = tccon_day['tccon_name'].values
            
            # Match each OCO-2 sounding
            for i, (lat_i, lon_i, time_i) in enumerate(zip(lat_oco, lon_oco, time_oco)):
                # Find TCCON measurements within spatial and temporal windows
                spatial_mask = (np.abs(lat_i - lat_tccon) < lat_lon_max) & \
                              (np.abs(lon_i - lon_tccon) < lat_lon_max)
                temporal_mask = np.abs(time_i - time_tccon) < time_max
                match_mask = spatial_mask & temporal_mask
                
                if np.any(match_mask):
                    # Get matching TCCON data
                    match_lat = lat_tccon[match_mask]
                    match_lon = lon_tccon[match_mask]
                    match_time = time_tccon[match_mask]
                    match_xco2 = xco2_tccon[match_mask]
                    match_name = name_tccon[match_mask]
                    
                    # Calculate distances
                    distances = [dist(lat_i, lat_j, lon_i, lon_j) for lat_j, lon_j in zip(match_lat, match_lon)]
                    
                    # Check if all matches are from the same station
                    unique_names = np.unique(match_name)
                    if len(unique_names) == 1:
                        # All matches from same station - use median
                        data_idx = data_day.index[i]
                        data.loc[data_idx, 'xco2tccon'] = np.nanmedian(match_xco2)
                        data.loc[data_idx, 'tccon_name'] = unique_names[0]
                        data.loc[data_idx, 'tccon_dist'] = np.nanmean(distances)
                    else:
                        # Multiple stations - use closest
                        closest_idx = np.argmin(distances)
                        data_idx = data_day.index[i]
                        data.loc[data_idx, 'xco2tccon'] = match_xco2[closest_idx]
                        data.loc[data_idx, 'tccon_name'] = match_name[closest_idx]
                        data.loc[data_idx, 'tccon_dist'] = distances[closest_idx]
    
    # Count matches
    n_matches = np.sum(data['xco2tccon'] > 0)
    print(f'Found {n_matches} TCCON matches out of {len(data)} soundings')
    
    return data


year = 2019
frac = 0.1 # fraction of data to load
name_all = 'LiteFileExport_' + str(year) + '_'
save_fig = True
Lite_file_path = '/Volumes/OCO/LiteFiles/export/B11.2_ML/'
save_path = '/Volumes/OCO/LiteFiles/B11.2v2ML_val/'

# check that we exported each file from B111 to B112
import os

# Paths to the two directories
OCO2_original_dir = '/Volumes/OCO/LiteFiles/B11.2_OCO2/'
OCO2_ML_dir = Lite_file_path

# Function to extract file identifiers from file names
def extract_identifier(filename):
    try:
        test = filename.split('.nc4')[1]
        filename = filename.split('_B11')[0]
        filename = filename.split('LtCO2_')[1]
    except:
        pass
    return filename

# List of file identifiers in each directory
files_in_dir1 = {extract_identifier(f) for f in os.listdir(OCO2_original_dir) if os.path.isfile(os.path.join(OCO2_original_dir, f))}
files_in_dir2 = {extract_identifier(f) for f in os.listdir(OCO2_ML_dir) if os.path.isfile(os.path.join(OCO2_ML_dir, f))}

# Identifiers unique to each directory
unique_to_dir1 = files_in_dir1 - files_in_dir2
unique_to_dir2 = files_in_dir2 - files_in_dir1

print("Files unique to B11.2_OCO2 directory:", unique_to_dir1)
print("Files unique to B11.2_ML directory:", unique_to_dir2)




# read in Lite files ************************************************************
# make path to save figures if it does not exist
if save_fig:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

fill_value = 999999
counts=0
print(year)
# get LiteFile data
Lite_files = glob.glob(Lite_file_path + 'oco2_LtCO2_' + str(year-2000) + '*.nc4')

# get Lite vars
l_ds = nc.Dataset(Lite_files[0])
l_vars, l_dims = get_all_headers_with_dims(l_ds)
# remove vars we dont need
vars_to_remove = [
    'bands', 'footprints', 'levels', 'vertices', 'Retrieval/iterations', 'file_index', 'vertex_latitude',
    'vertex_longitude', 'date', 'source_files', 'pressure_levels', 'co2_profile_apriori', 'xco2_averaging_kernel',
    'Preprocessors/co2_ratio_offset_per_footprint', 'Preprocessors/h2o_ratio_offset_per_footprint',
    'Retrieval/SigmaB', 'pressure_weight', 'xco2_qf_simple_bitflag', 'xco2_qf_bitflag',
    'Sounding/l1b_type', 'Sounding/orbit', 'frames',
]
l_vars = [e for e in l_vars if e not in vars_to_remove]
# remove any vars that start with 'L1b'
l_vars = [e for e in l_vars if not e.startswith('L1b')]

# Separate variables by dimension
l_vars_1d = []
l_vars_2d = []
l_vars_nd = []
for i, v in enumerate(l_vars):
    var = l_ds[v]
    ndim = var.ndim
    if ndim == 1:
        l_vars_1d.append(v)
    elif ndim == 2:
        l_vars_2d.append(v)
    else:
        l_vars_nd.append(v)

# Initialize data_dict keys
data_dict = {}
for v in l_vars_1d + l_vars_2d + l_vars_nd:
    data_dict[v] = []

# read in data
for j in tqdm(range(len(Lite_files))):
    l = Lite_files[j]
    l_ds = nc.Dataset(l)

    # Read 1D variables
    for v in l_vars_1d:
        val = l_ds[v][:]
        val = np.where(val == fill_value, np.nan, val)
        data_dict[v].append(val)

    # Read 2D variables
    for v in l_vars_2d:
        val = l_ds[v][:]  # shape: (num_soundings, dim2)
        val = np.where(val == fill_value, np.nan, val)
        data_dict[v].append(val)

    # Handle variables with ndim > 2
    for v in l_vars_nd:
        val = l_ds[v][:]  # shape: (num_soundings, dim2, dim3, ...)
        val = np.where(val == fill_value, np.nan, val)
        data_dict[v].append(val)

    counts += 1

# Concatenate data from all files
for v in data_dict.keys():
    data_dict[v] = np.concatenate(data_dict[v], axis=0)

# Create DataFrame
data = pd.DataFrame()

# Add 1D variables to DataFrame
for v in l_vars_1d:
    data[v] = data_dict[v]

# Add 2D variables to DataFrame (each entry is an array)
for v in l_vars_2d:
    data[v] = list(data_dict[v])

# Handle variables with ndim > 2 if needed (each entry is an array)
for v in l_vars_nd:
    data[v] = list(data_dict[v])

# downsample data to save RAM
interval = int(1 / frac)
data = data.iloc[::interval]
data['sounding_id'] = data['sounding_id'].astype(int)

# clean up features
Features = data.columns
feature_dict = {}
for f in Features:
    features_clean = f.split('/')[-1]
    if f.split('/')[0] != f.split('/')[-1]:
        feature_dict[f] = features_clean
data.rename(columns=feature_dict, inplace=True)

# Load and match TCCON data ****************************************************
print('Loading and matching TCCON data...')
tccon_data = load_tccon_data()
data = match_tccon_to_soundings(data, tccon_data)



with open(save_path +str(year)+ '_ML_bias_correction_stats.txt', 'w') as f:
    n_total = len(data)
    f.write('Total number of soundings: ' + str(n_total) + '\n')
    f.write('=' * 50 + '\n')


    for qf_version in ['xco2_quality_flag_ML', 'xco2_quality_flag']:
        f.write('Quality Flag var Used: ' + str(qf_version) + '\n')
        f.write('=' * 50 + '\n')
        
            
        # Define surface types to analyze
        surface_types = ['all', 'land', 'ocean']
        
        for surface in surface_types:
            f.write('\n--- ' + surface.title() + ' Surfaces ---\n')
            # Filter data based on surface type

            if surface == 'all':
                data_surface = data
            elif surface == 'land':
                data_surface = data[data['land_fraction'] == 100]
            elif surface == 'ocean':
                data_surface = data[data['land_fraction'] == 0]

            n_surface = len(data_surface)

            for qf in [0, 1, 2]:
                f.write('Quality Flag: ' + str(qf) + '\n')
                f.write('=' * 50 + '\n')

                # filter data based on quality flag
                data_surface = data_surface[data_surface[qf_version] == qf]
                
                n_surface_qf = len(data_surface)

                fraction_surface = n_surface_qf / n_surface if n_surface > 0 else 0
                fraction_total = n_surface_qf / n_total if n_total > 0 else 0
                
                f.write('Number of data points: ' + str(n_surface_qf) + '\n')
                f.write('Fraction of total data: ' + str(fraction_total) + '\n')
                f.write('Fraction of data per surface type: ' + str(fraction_surface) + '\n')
                f.write('=' * 50 + '\n')
                    
                # TCCON validation statistics for this surface
                data_tccon = data_surface[data_surface['xco2tccon'] > 0]  # Only soundings with TCCON matches
                
                if len(data_tccon) > 0:
                    if qf_version == 'xco2_quality_flag_ML':
                    # Calculate differences
                        diff = data_tccon['xco2_ML'] - data_tccon['xco2tccon']
                    else:
                        diff = data_tccon['xco2'] - data_tccon['xco2tccon']
                        diff_raw = data_tccon['xco2_raw'] - data_tccon['xco2tccon']
                        
                        rmse_raw = np.sqrt(np.nanmean(diff_raw**2))
                        std_raw = np.nanstd(diff_raw)
                        median_raw = np.nanmedian(diff_raw)
                    
                    # Calculate RMSE
                    rmse = np.sqrt(np.nanmean(diff**2))
                    # Calculate std and median
                    std = np.nanstd(diff)
                    median = np.nanmedian(diff)
                    
                    n_tccon = len(data_tccon)
                    
                    f.write('\nTCCON Validation:\n')
                    f.write('Number of TCCON matches: ' + str(n_tccon) + '\n')
                    if qf_version == 'xco2_quality_flag_ML':
                        f.write('xco2_ML_TCCON_RMSE: ' + str(rmse) + '\n')
                        f.write('xco2_ML_TCCON_std: ' + str(std) + '\n')
                        f.write('xco2_ML_TCCON_median: ' + str(median) + '\n')
                    else:
                        f.write('xco2_TCCON_RMSE: ' + str(rmse) + '\n')
                        f.write('xco2_TCCON_std: ' + str(std) + '\n')
                        f.write('xco2_TCCON_median: ' + str(median) + '\n')
                        f.write('xco2_TCCON_RMSE_raw: ' + str(rmse_raw) + '\n')
                        f.write('xco2_TCCON_std_raw: ' + str(std_raw) + '\n')
                        f.write('xco2_TCCON_median_raw: ' + str(median_raw) + '\n')
                else:
                    f.write('\nTCCON Validation:\n')
                    f.write('No TCCON matches found for this surface type.\n')
        
        f.write('\n' + '=' * 50 + '\n\n')


# visualize data ***************************************************************
data.loc[:,'ML-B11'] = data.loc[:,'xco2_ML'] - data.loc[:,'xco2']
data.loc[:,'ML-Raw'] = data.loc[:,'xco2_ML'] - data.loc[:,'xco2_raw']
data.loc[:,'B11-Raw'] = data.loc[:,'xco2'] - data.loc[:,'xco2_raw']
data_all = data.copy()
for qf in [0, 1, 2]:
    data = data_all.loc[data_all['xco2_quality_flag_ML'] == qf]
    name = name_all + 'QF' + str(qf)
    # Skip plotting if no data for this quality flag
    if len(data) == 0:
        print(f'No data for quality flag {qf}, skipping plots')
        continue
    plot_map(data, ['ML-Raw', 'ML-B11', 'B11-Raw'], save_fig=save_fig, path=save_path, name=name, pos_neg_IO=True, min=-1,max=1)
    plot_map(data, ['bias_correction_uncert_ML'], save_fig=save_fig, path=save_path, name=name, pos_neg_IO=False)




print('Done >>> ')
