# Steffen Mauceri & William Keely
# 02/2023

# Use a trained bias correction and filtering model,
# load a lite file, calc bias corrected XCO2, UQ of bias coorection, and Quality Flag,
# export the three vars to new Lite file


# make sure you have all of those packages installed in your conda envirnoment
import pandas as pd
import glob
import numpy as np
import netCDF4 as nc


from util import bias_correct, construct_filter 
import warnings
import os
from tqdm import tqdm
import paths
from pathlib import Path



# Make Changes *************************************************************************************
# Bias correction model directory paths
TC_LND_CORR_PATH = paths.TC_LND_CORR_MODEL
TC_OCN_CORR_PATH = paths.TC_OCN_CORR_MODEL
SA_LND_CORR_PATH = paths.SA_LND_CORR_MODEL
SA_OCN_CORR_PATH = paths.SA_OCN_CORR_MODEL

# filter model paths
TC_LND_FILT_PATH = paths.TC_LND_FILTER_MODEL
TC_OCN_FILT_PATH = paths.TC_OCN_FILTER_MODEL
SA_LND_FILT_PATH = paths.SA_LND_FILTER_MODEL
SA_OCN_FILT_PATH = paths.SA_OCN_FILTER_MODEL

# lite files paths
LITE_PATH = paths.OCO_LITE_FILES_DIR
EXPORT_LITE_PATH = paths.EXPORT_DIR / 'B11.2_ML'

# abstention filtering threshold on bias_correciton_uncert DEFAULT is 1.23 [ppm]
# value greater than 1.23 increases throughput in largely dusty regions such as N. Africa. Smaller values will remove more of the tropics.
ABSTENTION_THRESHOLD_LND = 1.3
ABSTENTION_THRESHOLD_OCN = 1.05

# Lite file name identifier (e.g. B11100Ar)
LITE_FILE_ID = 'B11210Ar'
# new Lite file name identifier after models are applied (e.g. B11Gamma)
New_LITE_FILE_ID = 'B112v2ML'

# debugging mode : only uses first 10 litefiles
COPY_LITE_FILES = True
DEBUG = False
# **************************************************************************************************
fill_value = -999999


# supress sklearn UserWarning and Pandas warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None

def get_all_headers(f):
    headers = list(f.variables.keys())

    # get groups
    groups = list(f.groups.keys())

    for g in groups:
        vars = list(f[g].variables.keys())
        for v in vars:
            headers.append(g+ '/' + v )

    return headers

def pre_filter(data):

    assert 'xco2_quality_flag_ML' in data.columns, 'xco2_quality_flag_ML not in data.columns'

    # filter data
    drop = np.zeros_like(data['xco2'], dtype=bool)

    drop[data['h2o_ratio_bc'] >= 1.1] = True
    drop[(data['co2_grad_del'] <= -100)] = True
    drop[(data['co2_grad_del'] >= 100)] = True

    #land
    drop[(data['snow_flag'] == 1) & (data['land_fraction'] == 100)] = True  # remove snow
    drop[(data['co2_ratio_bc'] >= 1.04) & (data['land_fraction'] == 100)] = True
    drop[(data['co2_ratio_bc'] <= 0.99) & (data['land_fraction'] == 100)] = True

    drop[(data['dpfrac'] >= 5) & (data['land_fraction'] == 100)] = True
    drop[(data['dpfrac'] <= -7.5) & (data['land_fraction'] == 100)] = True
    drop[(data['aod_ice'] >= 0.2) & (data['land_fraction'] == 100)] = True
    drop[(data['dws'] >= 1.0) & (data['land_fraction'] == 100)] = True

    drop[(data['dust_height'] <= 0.7) & (data['land_fraction'] == 100)] = True
    drop[(data['dust_height'] >= 1.75) & (data['land_fraction'] == 100)] = True
    drop[(data['rms_rel_sco2'] >= 1) & (data['land_fraction'] == 100)] = True
    drop[(data['snr_sco2'] <= 50) & (data['land_fraction'] == 100)] = True

    drop[(data['max_declocking_wco2'] >= 1.5) & (data['land_fraction'] == 100)] = True
    drop[(data['h_continuum_wco2'] >= 100) & (data['land_fraction'] == 100)] = True
    drop[(data['deltaT'] >= 1.5) & (data['land_fraction'] == 100)] = True
    drop[(data['albedo_slope_wco2'] >= 0) & (data['land_fraction'] == 100)] = True
    drop[(data['albedo_quad_sco2'] >= 0.000005) & (data['land_fraction'] == 100)] = True
    drop[(data['albedo_quad_sco2'] <= -0.000005) & (data['land_fraction'] == 100)] = True
    #sea
    drop[(data['co2_ratio_bc'] <= 0.99) & (data['land_fraction'] == 0)] = True
    drop[(data['co2_ratio_bc'] >= 1.03) & (data['land_fraction'] == 0)] = True
    drop[(data['deltaT'] <= -0.8) & (data['land_fraction'] == 0)] = True
    drop[(data['rms_rel_sco2'] >= 0.5) & (data['land_fraction'] == 0)] = True
    drop[(data['snr_sco2'] <= 200) & (data['land_fraction'] == 0)] = True
    drop[(data['max_declocking_wco2'] >= 0.75) & (data['land_fraction'] == 0)] = True

    # set values of data.xco2_quality_flag_ML to 2 where drop is True
    data.loc[drop, 'xco2_quality_flag_ML'] = 2

    return data

# make EXPORT_LITE_PATH if it does not exist
paths.ensure_dir_exists(EXPORT_LITE_PATH)


# copy files from LITE_PATH to EXPORT_LITE_PATH
lite_files = sorted(list(LITE_PATH.glob('*oco2_LtCO2*.nc4')))
lite_files.sort()

if DEBUG:
    lite_files = lite_files[:5]
    print('***************** WE ARE IN DEBUG MODE *****************')

if COPY_LITE_FILES:
    print('copying files ...')
    for l in lite_files:
        l_name = l.name
        print(l_name)
        # copy file
        os.system(f'cp "{l}" "{EXPORT_LITE_PATH / l_name}"')



# itterate over all lite files
for j in tqdm(range(len(lite_files))):
    
    # get LiteFile data
    l = lite_files[j]
    l_name = l.name
    print(l_name)
    l_ds = nc.Dataset(str(l))
    l_ids = l_ds['sounding_id'][:]
    
    # remove vars we dont need
    l_vars = get_all_headers(l_ds)
    l_vars = [e for e in l_vars if e not in ('bands', 'footprints','levels', 'vertices','Retrieval/iterations','file_index','vertex_latitude','vertex_longitude',
                                             'date','source_files','pressure_levels', 'co2_profile_apriori', 'xco2_averaging_kernel','Preprocessors/co2_ratio_offset_per_footprint',
                                             'Preprocessors/h2o_ratio_offset_per_footprint','Retrieval/SigmaB', 'frames',
                                             'L1b/land_fraction', 'L1b/latitude','L1b/longitude', 'L1b/operation_mode','L1b/orbit', 'L1b/selection_flag_sel', 'L1b/sounding_l1b_quality_flag', 'L1b/time',
                                             'pressure_weight', 'xco2_qf_simple_bitflag', 'xco2_qf_bitflag', 'Sounding/l1b_type', 'Sounding/orbit')]    


    data = np.ones((len(l_ids), len(l_vars))) * fill_value

    # load values of features and XCO2
    i = -1
    for v in l_vars:
        i += 1
        data[:, i] = l_ds[v][:]

    l_ds.close()
    data = pd.DataFrame(data, columns=l_vars)

    # Remove group names from variable names
    feature_dict = {}
    for f in data.columns:
        features_clean = f.split('/')[-1]
        if f.split('/')[0] != f.split('/')[-1]:
            feature_dict[f] = features_clean
    data.rename(columns=feature_dict, inplace=True)
    

    # perfrom bias correction
    print('Performing bias correction ... ')

    # split data into land and sea
    data_lnd = data[(data['land_water_indicator'] == 0) | (data['land_water_indicator'] == 3)]
    data_sea = data[(data['land_water_indicator'] == 1)]



    # Perform bias correction -------------------------------------------------------------------------------
    if len(data_lnd) > 0:
        data_lnd = bias_correct(TC_LND_CORR_PATH, data_lnd, ['xco2_raw'], uq = True)
        data_lnd = bias_correct(SA_LND_CORR_PATH, data_lnd, ['xco2_raw'], uq = False)
    if len(data_sea) > 0:
        data_sea = bias_correct(TC_OCN_CORR_PATH, data_sea, ['xco2_raw'], uq = True)
        data_sea = bias_correct(SA_OCN_CORR_PATH, data_sea, ['xco2_raw'], uq = False)

        # adjust sea based on land-ocean crossings
        data_sea['xco2_raw'] = data_sea['xco2_raw'] - 0.15

    # combine data
    if len(data_lnd) > 0 and len(data_sea) > 0:
        data = pd.concat([data_lnd, data_sea])
    elif len(data_lnd) > 0:
        data = data_lnd
    elif len(data_sea) > 0:
        data = data_sea
    # sort data by sounding id
    data.sort_values('sounding_id', inplace=True)



    # Perform quality filtering -------------------------------------------------------------------------------
    data.loc[:,'xco2_ML'] = data.loc[:,'xco2_raw']
    print('data.shape: ', data.shape)
    print('Adding ML quality filter flag ... ')
    #  path dict and abstention threshold
    kwargs = {
        'path_tc_lnd' : TC_LND_FILT_PATH,
        'path_tc_ocn' : TC_OCN_FILT_PATH,
        'path_sa_lnd' : SA_LND_FILT_PATH,
        'path_sa_ocn' : SA_OCN_FILT_PATH,
        'abstention_threshold_lnd' : ABSTENTION_THRESHOLD_LND,
        'abstention_threshold_ocn' : ABSTENTION_THRESHOLD_OCN
    }

    #  add ternary flag
    data = construct_filter(data, **kwargs)

    # sets 'pre-filtered' data to '2' in xco2_quality_flag_b112
    data = pre_filter(data)



    # Export data to new Lite file -------------------------------------------------------------------------------
    print('data.shape after adding filter : ', data.shape)
    ## add bias corrected values to LiteFiles in export folder

    export_file_path = EXPORT_LITE_PATH / l_name # Construct the path first
    # Use exists() instead of globbing for a single known file
    # export_file = list((EXPORT_LITE_PATH / l_name).parent.glob(l_name))
    print('Export file path: ', export_file_path)
    # if len(export_file) == 1:
    if export_file_path.exists():
        # open the netCDF4 file in append mode
        ds = nc.Dataset(str(export_file_path), 'a')
        if 'xco2_ML' not in ds.variables.keys():
            value = ds.createVariable('xco2_ML', np.float32, 'sounding_id')
            value.units = 'ppm'
            value.long_name = 'XCO2 Machine Learning corrected'
            value.comment = 'Column-averaged dry-air mole fraction of CO2, including bias correction, on X2019 scale, based on ACOS v11.2. Further described in Mauceri et. al. 2025 "Uncertainty-aware Machine Learning Bias Correction and Filtering for OCO-2: Part 1" https://doi.org/10.1029/2025EA004328'
            value[:] = data.loc[:,'xco2_ML'].to_numpy().astype(np.float32)
            # add global attribute of what model we used for bias correction
            # Use the path variables directly
            ds.setncattr('xco2_bias_correction_ML_model', f'{TC_LND_CORR_PATH.name} and '
                         f'{TC_OCN_CORR_PATH.name} and '
                         f'{SA_LND_CORR_PATH.name} and '
                         f'{SA_OCN_CORR_PATH.name}')

        if 'xco2_quality_flag_ML' not in ds.variables.keys():
            value = ds.createVariable('xco2_quality_flag_ML', np.intc, 'sounding_id')
            value.units = 'ternary : 0 = best quality data, 1 = good quality data for increasing sounding throughput if needed, 2 = poor quality data'
            value.long_name = 'XCO2 ternary quality flag'
            value.comment = 'Derived from two machine learning models and uncertainty of the bias correction estimate. Further described in Keely et. al. 2025 "Uncertainty-aware Machine Learning Bias Correction and Filtering for OCO-2: Part 2" https://doi.org/10.1029/2025EA004329'
            value[:] = data.loc[:,'xco2_quality_flag_ML'].to_numpy().astype(np.float32)
            # add global attribute of what model we used for quality filtering
            # Use the path variables directly
            ds.setncattr('xco2_quality_flag_ML model', f'{TC_LND_FILT_PATH.name} and '
                         f'{TC_OCN_FILT_PATH.name} and '
                         f'{SA_LND_FILT_PATH.name} and '
                         f'{SA_OCN_FILT_PATH.name}')
            # add global attribute of what abstention thresholds we used for quality filtering
            ds.setncattr('abstention_threshold_lnd', ABSTENTION_THRESHOLD_LND )
            ds.setncattr('abstention_threshold_ocn', ABSTENTION_THRESHOLD_OCN )

        if 'bias_correction_uncert_ML' not in ds.variables.keys():
            value = ds.createVariable('bias_correction_uncert_ML', np.float32, 'sounding_id')
            value.units = 'ppm'
            value.long_name = 'XCO2 bias correction uncertainty'
            value.comment = '1-simga uncertainty of the ML bias correction model.'
            value[:] = data.loc[:,'bias_correction_uncert'].to_numpy().astype(np.float32)

        # overwrite some attribute
        ds.setncattr('identifier_product_doi', 'Unassigned')
        ds.setncattr('contact', 'Steffen Mauceri: Steffen.Mauceri@jpl.nasa.gov')
        ds.setncattr('title', 'OCO-2 Lite Files with ML bias correction and filtering')
        ds.setncattr('creation_date', pd.Timestamp.now().date().strftime('%Y-%m-%d'))

        # rename global attributes
        ds.renameAttribute('filter_function', 'filter_function_xco2')
        ds.setncattr('bc_function', 'bc_function_xco2')

        ds.close()
    else:
        # raise ValueError(l_name + ' not available in export folder')
        raise FileNotFoundError(f'{export_file_path} not found in export folder')
        
    # renmame files in export folder
    old_name = EXPORT_LITE_PATH / l_name
    new_name = EXPORT_LITE_PATH / (l_name.replace(LITE_FILE_ID, New_LITE_FILE_ID))
    # remove everything after last '_' in new_name
    new_name = new_name.parent / (new_name.name[:new_name.name.rfind('_')] + '.nc4')
    os.rename(old_name, new_name)


print('Done >>>')