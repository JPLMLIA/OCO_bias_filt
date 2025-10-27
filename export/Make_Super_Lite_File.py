import netCDF4 as nc
import glob
import os
import numpy as np
from tqdm import tqdm

# Folder containing the Lite Files
INPUT_FOLDER = '/Volumes/OCO/LiteFiles/B11.2_ML'
# Path for the output combined file
OUTPUT_FILE = '/Volumes/OCO/LiteFiles/B11.2_ML/SuperLite_B11.2_ML.nc4'

# List of variables to keep (all other variables will be discarded)
KEEP_VARIABLES = [
    'sounding_id',
    'xco2_ML', 'xco2', 'xco2_x2019',
    'xco2_quality_flag_ML', 'xco2_quality_flag',
    'bias_correction_uncert_ML', 'xco2_uncertainty',
    'latitude', 'longitude',
    'time',
    'Sounding/land_water_indicator', 'Sounding/operation_mode',
]


def read_netcdf_variables(file_path, variables):
    """
    Open a netCDF file and read only the specified variables.
    """
    ds = nc.Dataset(file_path, 'r')
    data = {}
    for var in variables:
        if '/' in var:
            group_name, var_name = var.split('/', 1)
            if group_name in ds.groups and var_name in ds.groups[group_name].variables:
                data[var] = ds.groups[group_name].variables[var_name][:].copy()
            else:
                print(f"Warning: Variable {var} not found in file {file_path}.")
        else:
            if var in ds.variables:
                data[var] = ds.variables[var][:].copy()
            else:
                print(f"Warning: Variable {var} not found in file {file_path}.")
    ds.close()
    return data


def combine_data(data_list, variables):
    """
    Combine the data for each variable along the first dimension.
    Assumes that each variable is a 1D array and that the dimension represents sounding_id.
    """
    combined = {}
    for var in variables:
        # Only include data arrays that exist for the given variable
        arrays = [data[var] for data in data_list if var in data]
        if arrays:
            combined[var] = np.concatenate(arrays)
        else:
            print(f"Error: No data found for variable **{var}**.")
            combined[var] = np.array([])
    return combined


def write_combined_netcdf(output_file, combined_data, source_file):
    """
    Create a new netCDF file, write the combined data, copy global attributes,
    and copy variable attributes from the source file.
    """
    # Open the source file and get its global and variable attributes
    src = nc.Dataset(source_file, 'r')
    global_attrs = {attr: src.getncattr(attr) for attr in src.ncattrs()}
    # Store variable attributes for variables that exist in both files
    var_attrs = {}
    for var in combined_data.keys():
        if '/' in var:
            group_name, var_name = var.split('/', 1)
            if group_name in src.groups and var_name in src.groups[group_name].variables:
                var_attrs[var] = {attr: src.groups[group_name].variables[var_name].getncattr(attr)
                                  for attr in src.groups[group_name].variables[var_name].ncattrs()}
        else:
            if var in src.variables:
                var_attrs[var] = {attr: src.variables[var].getncattr(attr)
                                  for attr in src.variables[var].ncattrs()}
    src.close()

    # Create the output dataset
    ds_out = nc.Dataset(output_file, 'w', format='NETCDF4')

    # Copy the global attributes
    for attr, value in global_attrs.items():
         ds_out.setncattr(attr, value)

    #TODO Update DOI once assigned
    ds_out.setncattr('identifier_product_doi', 'Unassigned')

    # Create the 'sounding_id' dimension
    num_records = combined_data['sounding_id'].shape[0]
    ds_out.createDimension('sounding_id', num_records)

    # Create and write each variable along with its attributes (flattening group variables)
    for var, data in combined_data.items():
         # Flatten the variable name by using only the part after '/' if it exists
         var_name = var.split('/')[-1] if '/' in var else var
         # Determine dimensions for the variable. The first dimension is always 'sounding_id'.
         dims = ['sounding_id']
         # If the variable has additional dimensions, create them with unique names
         if data.ndim > 1:
              for i in range(1, data.ndim):
                  dim_name = f"{var_name}_dim_{i}"
                  dims.append(dim_name)
                  # Create the extra dimension if it doesn't exist already
                  if dim_name not in ds_out.dimensions:
                       ds_out.createDimension(dim_name, data.shape[i])
         var_out = ds_out.createVariable(var_name, data.dtype, tuple(dims))
         var_out[:] = data
         # Copy variable attributes if available
         if var in var_attrs:
             for attr, value in var_attrs[var].items():
                 var_out.setncattr(attr, value)

    ds_out.close()


def main():

    for year in range(2014, 2026):
        year = str(year)

        # Find all netCDF Lite Files in the input folder
        file_paths = sorted(glob.glob(os.path.join(INPUT_FOLDER, f'*LtCO2_{year[-2:]}*.nc4')))
        if not file_paths:
            print("No Lite Files found in folder:", INPUT_FOLDER)
            continue

        all_data = []
        for file in tqdm(file_paths):
            # print("Processing file:", file)
            file_data = read_netcdf_variables(file, KEEP_VARIABLES)
            all_data.append(file_data)

        combined_data = combine_data(all_data, KEEP_VARIABLES)

        output_file_yr = OUTPUT_FILE.replace('SuperLite', f'SuperLite_{year}')
        write_combined_netcdf(output_file_yr, combined_data, file_paths[0])
        print("Combined Lite File created:", output_file_yr)
        # get size of the file
        file_size = os.path.getsize(output_file_yr) / 1024 / 1024
        print(f"File size: {file_size:.2f} MB")

    print('Done >>>')


if __name__ == '__main__':
    main()
