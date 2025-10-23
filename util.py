# util functions
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent automatic display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from scipy.odr import Model, ODR, RealData
# import forestci as fci
from tqdm import tqdm
import collections
import cartopy.crs as ccrs
from pathlib import Path # Add Path
import json # Ensure json is imported
import joblib # Ensure joblib is imported

# Import paths from the root directory
import paths




def normalize_per_SA(data, vars, add=False):
    '''
    Normalize data per Small Area
    :param data: pd.DataFrame, data
    :param vars: list, variables to normalize
    :param add: bool, add new features instead of overwriting existing features
    :return: pd.DataFrame, normalized data
    '''
    print('normalizing data per SA')
    # process one cluster at a time
    SA = list(pd.unique(data['SA']))
    idx_all = np.arange(0, len(data))
    data_SA = data['SA'].to_numpy()

    if add:
        # add new features instead of overwriting existing featues
        vars_norm = vars.copy()
        for i in range(len(vars_norm)):
            vars_norm[i] = vars_norm[i] + '_norm'

    data_vars = data[vars].to_numpy()
    for a in SA:
        # get indexes that belong to SA
        idx = idx_all[data_SA == a]
        # remove median from features per SA
        data_vars[idx, :] = data_vars[idx, :] - np.median(data_vars[idx, :])

    if add:
        data[vars_norm] = pd.DataFrame(data_vars, index=data.index)
        return data, vars_norm
    else:
        data[vars] = data_vars
        return data


def make_prediction(M, X, model, UQ=False, X_train=0, y_mean=0, y_std=1):
    '''
    Predict the bias using a trained model
    :param M: model 
    :param X: data
    :param model: model
    :param UQ: uncertainty
    :param X_train: training data
    :param y_mean: mean of y
    :param y_std: std of y
    :return: bias, bias_std
    '''
    if model == 'RF':
        bias = M.predict(X)
        if UQ:
            # M.max_samples = 100000
            # bias_std = fci.random_forest_error(M, X_train=X_train.values, X_test=X.values, memory_constrained=True, memory_limit=50000) ** 0.5
            df_temp = pd.DataFrame()
            for pred in M.estimators_:
                temp = pd.Series(pred.predict(X))
                df_temp = pd.concat([df_temp, temp], axis=1)
            std_temp = pd.DataFrame()
            for percentile in [0.16, 0.84]: # 0.16 and 0.84 percentiles corespond to a 1-sigma prediction interval
                p = df_temp.quantile(q=percentile, axis=1)
                std_temp = pd.concat([std_temp, p], axis=1)
            std_temp.columns = ['lower', 'upper']
            bias_std = np.abs(std_temp['upper'] - std_temp['lower'])
            bias_std = bias_std.to_numpy()


        else:
            bias_std = bias

        return bias, bias_std

    if model == 'XGB':
        bias = M.predict(X)

        return bias, bias

    if model == 'GPR':
        if UQ:
            bias, bias_std = M.predict(X, return_std=True)
            bias_std = bias_std * y_std + y_mean
        else:
            # chop prediction into 10000 samples at a time
            n = 100000
            bias = np.zeros(len(X))
            for i in range(0, len(X), n):
                bias[i:i + n] = M.predict(X[i:i + n], return_std=False)
            bias_std = bias
        # reverse normalization
        bias = bias * y_std + y_mean
        return bias, bias_std

    if model == 'Ridge':
        bias = M.predict(X)
        # reverse normalization
        bias = bias * y_std + y_mean
        return bias, bias

    if model == 'Ransac':
        bias = M.predict(X)
        # reverse normalization
        bias = bias * y_std + y_mean
        return bias, bias

    if model == 'NN':
        bias = M.predict(X)
        # reverse normalization
        bias = bias * y_std + y_mean
        return bias, bias


def remove_missing_values(data):
    '''
    Remove missing values
    :param data: pd.DataFrame, data
    :return: pd.DataFrame, data with missing values removed
    '''
    print('removing missing values')

    data.replace(-999999, np.nan, inplace=True)
    data.replace(np.inf, np.nan, inplace=True)

    data['xco2_raw'].replace(0, np.nan, inplace=True)
    data['xco2'].replace(0, np.nan, inplace=True)

    # make an expception for TCCON and Model data and cld data
    vars = ['CT_2022+NRT2023-1', 'LoFI_m2ccv1bsim', 'MACC_v21r1', 'UnivEd_v5.2', 'xco2tccon',
            'tccon_name','tccon_dist','tccon_xluft', 'tccon_xluft_running','tccon_airmass','tccon_xh2o','tccon_pout','cld_dist']

    for v in vars:
        if v in data:
            data[v].replace(np.nan, 0, inplace=True)

    # remove rows with nans
    data.dropna(inplace=True)

    #change TCCON and Models back to nan
    for v in vars:
        if v in data:
            data[v].replace(0, np.nan, inplace=True)

    return data


def remove_SA_by_size(data, min_SA_size, verbose_IO):
    '''
    Remove Small Areas with too view samples
    :param data: pd.DataFrame, data
    :param min_SA_size: int, minimum number of samples per SA
    :param verbose_IO: bool, verbose output
    :return: pd.DataFrame, data with SA by size removed
    '''
    # remove small areas with too view samples
    print('removing SA by size')
    data_SA = data['SA'].to_numpy()
    idx_all = data.index

    SA, counts = np.unique(data_SA, return_counts=True)

    SA_too_small = SA[counts <= min_SA_size]
    idx_drop=[]
    for a in SA_too_small:
        idx_drop.append(idx_all[data_SA == a])

    if len(idx_drop) > 1:
        idx_drop = np.concatenate(idx_drop)
        data.drop(index=idx_drop, inplace=True)

    if verbose_IO:
        plt.figure()
        plt.hist(counts[::100], bins=20)
        plt.vlines(min_SA_size, 0, 20, colors='r')
        plt.show()

    return data


def filt_corr_TCCON(data, filt_TCCON, correct_TCCON):
    '''filter and bias correct TCCON data
    :param data: pd.DataFrame; data
    :param filt_TCCON: bool; filter TCCON data
    :param correct_TCCON: bool; correct TCCON data
    :return: pd.DataFrame; filtered and bias corrected data
    '''
    # filter TCCON data
    if filt_TCCON:
        data = data.loc[((data['tccon_xluft_running'] - 0.999).abs() < filt_TCCON) | np.isnan(data['tccon_xluft_running']), :]

    if correct_TCCON:
        # # Josh Bias correction
        corr_var = 'tccon_xluft_running'  # 'tccon_xluft_running'
        data['xco2tccon'] = data['xco2tccon'] / (0.356 * data[corr_var] + 0.644)

    return data


def load_data(year, mode, min_SA_size=20, verbose_IO=False, qf=None, preload_IO = True, clean_IO=True, footprint=0,
            TCCON=False, balanced=False, Save_RAM=False, remove_inland_water=False, max_n = 5 * 10 ** 6):
    '''
    Load soundings from pkl file
    :param year: int, year
    :param mode: str, mode
    :param min_SA_size: int, minimum number of samples per SA
    :param verbose_IO: bool, verbose output
    :param qf: int, quality flag
    :param preload_IO: bool, preload data
    :param clean_IO: bool, clean data
    :param footprint: int, footprint
    :param TCCON: bool, TCCON data
    :param balanced: bool, balanced data
    :param Save_RAM: bool, save RAM
    :param remove_inland_water: bool, remove inland water
    :param max_n: int, maximum number of samples
    :return: pd.DataFrame, data
    '''

    if preload_IO & clean_IO:
        # max_n = 10**7 # to not run out of RAM

        # Get the path using the new function from paths.py
        path = paths.get_preload_filepath(mode, year)

        print(f"Load preloaded file: {path}")
        data = pd.read_parquet(path)

        if Save_RAM:
            #drop every 2nd sample to save RAM
            data = data.iloc[::2]

        if TCCON: # remove soundings without TCCON match ups
            data = data[data['xco2tccon'] > 0]

        if remove_inland_water: # remove inland water
            data = data.loc[~((data['land_water_indicator'] == 1) & (data['altitude'] != 0)), :]

    else:

        # Construct path for filtered files using paths.py
        filtered_filename = f'LiteB112v2_filt_{year}.parquet' # Changed extension from .pkl to .parquet
        path = paths.PAR_DIR / filtered_filename
        print(f"Load filtered file: {path}")
        data = pd.read_parquet(path)

        # remove 'xco2' column
        data.drop(columns=['xco2'], inplace=True)
        # replace xco2 with xco2_x2019
        data.rename(columns={'xco2_x2019': 'xco2'}, inplace=True)

        if Save_RAM:
            #drop every 2nd sample
            data = data.iloc[::2]

        # remove data by quality flag
        if qf is not None:
            data = data.loc[data['xco2_quality_flag'] == qf]

        # remove sea or land data
        # land_water_indicator: 0: land; 1: water; 2: inland water; 3: mixed.
        # operation_mode: Nadir(0), Glint(1), Target(2), or Transition(3)
        if mode == 'LndNDGL':
            print('removing ocean')
            data = data.loc[(data['land_water_indicator'] == 0), :]
            data = data.loc[(data['operation_mode'] != 3), :]
            data.drop(columns=['windspeed', 'windspeed_apriori'], inplace=True)

        elif mode == 'LndND':
            print('removing ocean')
            data = data.loc[(data['land_water_indicator'] == 0), :]
            data = data.loc[data['operation_mode'] == 0, :]
            data.drop(columns=['windspeed', 'windspeed_apriori'], inplace=True)

        elif mode == 'LndGL':
            print('removing ocean')
            data = data.loc[(data['land_water_indicator'] == 0), :]
            data = data.loc[data['operation_mode'] == 1, :]
            data.drop(columns=['windspeed', 'windspeed_apriori'], inplace=True)

        elif mode == 'SeaGL':
            print('removing land')
            data = data.loc[(data['land_water_indicator'] == 1), :]

        elif mode == 'all':
            print('removing nothing')
            data = data.loc[(data['operation_mode'] != 3), :]
            data.drop(columns=['windspeed', 'windspeed_apriori'], inplace=True)

        else:
            print('mode needs to be [LndNDGL, LndND, LndGL, SeaGL, all] ')


        if clean_IO: # clean data
            # remove missing values
            data = remove_missing_values(data)

            data = data.loc[data['xco2_MLquality_flag'] == 0, :]

            # remove TCCON data that is too far from OCO-2 observations
            data = data.loc[data['tccon_dist_quality_flag'] != 1, :]

            # set SA of soundings that are close to strong CO2 emitters to nan
            # print('removing soundings that are close to strong CO2 emitters')
            data.loc[data['strong_emitter'] == 1, 'SA'] = np.nan

            # remove small areas with too view samples
            #data = remove_SA_by_size(data, min_SA_size, verbose_IO)

        if footprint > 0:
            data = data[data['footprint'] == footprint]

        if len(data) > max_n:
            # make sure we get all soundings with TCCON match ups
            data_t = data.loc[data['xco2tccon'] > 0]

            if balanced:
                # weight SAs by geographic density
                data_not_nan = data.dropna(subset=['SA'])
                weights = balance_sounding_loc(data_not_nan) # 'data' might have NaNs in 'SA' if we removed 'strong_emitter' above

                # subsample SA's from data until we have at least max_n samples - len(data_t)
                SAs = sorted(data_not_nan['SA'].unique())
                previous_SAs = []
                data_sampled = []
                n_samples = 0
                data_grouped = data.groupby('SA')
                print('sampling SAs')
                SA_is = np.random.choice(SAs, size=len(SAs), replace=False, p=weights)
                i = 0
                while n_samples < max_n - len(data_t):
                    SA_i = SA_is[i]
                    i += 1
                    # sample a SA with weights
                    previous_SAs.append(SA_i)
                    # get the data for the SA
                    data_SA = data_grouped.get_group(SA_i)
                    n_samples += len(data_SA)
                    # append to data_sampled
                    data_sampled.append(data_SA)
                    # make sure we break if we can't sample any more SAs
                    if len(previous_SAs) == len(SAs)-10:
                        break
                # concat data_sampled to dataFrame
                data_SA = pd.concat(data_sampled)
            else:
                # use this if we don't want to weight SAs
                data_SA = data.sample(max_n - len(data_t), random_state=1)

            # concat data_SA and data_t
            data = pd.concat([data_SA, data_t])

    return data

def balance_sounding_loc(data):
    ''' calculates weights for SA of data depending on density of soundings for a given location
    :param data: DataFrame
    :return: ndarray; of weights that can be used to subsample data
    '''
    print('calculating weights to balance soundings ...')

    res = 2 # in degrees

    # get mean lat and lon for each SA with groupby
    Lat = data.groupby('SA')['latitude'].mean().values
    Lon = data.groupby('SA')['longitude'].mean().values
    SA = data.groupby('SA')['SA'].mean().values
    weights = np.zeros((len(SA)))

    for i in tqdm(range(len(np.arange(-90, 90, res)))):
        Lat_i = np.arange(-90, 90, res)[i]
        for Lon_i in range(-180, 180, res):
            # check if we have measurements in bin
            match = (Lat >= Lat_i) & (Lat < Lat_i + res) & (Lon >= Lon_i) & (Lon < Lon_i + res)
            bin = np.sum(match)
            if bin > 0:
                weights[match] = 1/bin
            else:
                weights[match] = 0

    # make sure all weights sum to 1
    weights = weights / np.sum(weights)
    return weights


def scatter_density(x, y, x_name, y_name, title, dir, save_IO=False):
    '''
    makes a scatter plot and color codes where most of the data is

    :param x: x-value
    :param y: y-value
    :param x_name: name on x-axis
    :param y_name: name on y-axis
    :param title: title
    :param dir: save location
    :param save_IO: save plot?
    :return: -
    '''
    
    print('making scatter density plot ... ')
    # need to reduce number of samples to keep processing time reasonable.
    # Reduce if processing time too long or run out of RAM
    max_n = 50000
    if len(x) > max_n:
        subsample = int(len(x) / max_n)
        x = x[::subsample]
        y = y[::subsample]
    try:
        r, _ = stats.pearsonr(x, y) # get R
    except:
        print('could not calculate r, set to nan')
        r = np.nan
    xy = np.vstack([x, y])
    if np.mean(x) == np.mean(y):
        z = np.arange(len(x))
    else:
        z = stats.gaussian_kde(xy)(xy)# calculate density
    # sort points by density
    idx = z.argsort()
    d_feature = x[idx]
    d_target = y[idx]
    z = z[idx]
    # plot everything
    plt.scatter(d_feature, d_target, c=z, s=2, label='R = ' + str(np.round(r, 2)))
    plt.legend()
    plt.xlim(np.percentile(d_feature, 1), np.percentile(d_feature, 99))
    plt.ylim(np.percentile(d_target, 1), np.percentile(d_target, 99))
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    if save_IO:
        paths.ensure_dir_exists(dir)
        plt.savefig(dir / (title + '.png'), dpi=300)
        
    else:
        plt.show()
    plt.close()

def scatter_3D(x, y, z, x_name, y_name, title, save_path: Path, save_IO=False):
    '''
    Make a 3D scatter plot
    :param x: x-value
    :param y: y-value
    :param z: z-value
    :param x_name: name on x-axis
    :param y_name: name on y-axis
    :param title: title
    :param save_path: save location
    :param save_IO: save plot?
    :return: -
    '''
    fig = plt.figure(figsize=(10, 10))
    max_n = 10000
    if len(x) > max_n:
        subsample = int(len(x) / max_n)
        x = x[::subsample]
        y = y[::subsample]
        z = z[::subsample]

    # idx = z.argsort()
    d_feature = x  # [idx]
    d_target = y  # [idx]
    # z = z[idx]

    plt.scatter(d_feature, d_target, c=z, s=0.1, cmap='viridis', vmin=np.percentile(z, 5), vmax=np.percentile(z, 95))
    plt.xlim(np.percentile(d_feature, 2), np.percentile(d_feature, 98))
    plt.ylim(np.percentile(d_target, 2), np.percentile(d_target, 98))
    plt.colorbar()
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.title(title)
    plt.tight_layout()
    if save_IO:
        paths.ensure_dir_exists(save_path.parent)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def scatter_hist(x_var, y_var, x_name, y_name, title, save_path: Path, save_IO=False, bias_IO=True):
    '''
    Make a scatter plot and histogram
    :param x_var: x-value
    :param y_var: y-value
    :param x_name: name on x-axis
    :param y_name: name on y-axis
    :param title: title
    :param save_path: save location
    '''
    nbins = 50
    # bin error by variable
    bin_mean = np.zeros(nbins) * np.nan
    bin_5 = np.zeros(nbins) * np.nan
    bin_95 = np.zeros(nbins) * np.nan
    bin_edges = np.linspace(np.percentile(x_var,1), np.percentile(x_var,99), nbins+1)
    # bin_edges = np.linspace(np.min(x_var), np.max(x_var), nbins + 1)
    #_, bin_edges = np.histogram(x_var, bins=nbins)
    gap = np.mean(np.diff(bin_edges))
    x = bin_edges[:-1] + gap / 2
    x[0] = bin_edges[0]
    x[-1] = bin_edges[-1]

    i = -1
    for bin in bin_edges[:-1]:
        i += 1
        t = y_var[(x_var >= bin) & (x_var < bin + gap)]
        if len(t) > 0:
            bin_mean[i] = np.mean(t)
            bin_5[i] = np.percentile(t, 5)
            bin_95[i] = np.percentile(t, 95)

    plt.figure(figsize=(4, 3))
    plt.scatter(x_var, y_var, s=1, color='gray', zorder=1)
    plt.fill_between(x, bin_5, bin_95, color='orange', zorder=2, alpha=0.5)
    plt.plot(x, bin_mean, color='red')
    plt.xlim(x[0], x[-1])

    if bias_IO:
        plt.ylim(-2, 2)
        #plt.ylim(-5, 5)
    else:
        plt.ylim(-5, 2)

    plt.grid()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    if save_IO:
        paths.ensure_dir_exists(save_path.parent)
        plt.savefig(save_path, dpi=300)
    plt.close()


def get_RMSE(data, ignore_nan=False):
    '''
    Calculate the RMSE
    :param data: np.array, data
    :param ignore_nan: bool, ignore nan
    :return: float, RMSE
    '''
    if ignore_nan:
        RMSE = np.nanmean(data ** 2) ** 0.5
    else:
        RMSE = np.mean(data ** 2) ** 0.5

    return RMSE


def get_variability_reduction(data, var_tp, name, path: Path, save_fig=False, qf=None):
    '''
    Calculate the variability reduction in small areas
    :param data: pd.DataFrame, data
    :param var_tp: str, variable type
    :param name: str, name
    :param path: Path, save location
    :param save_fig: bool, save figure
    :param qf: int, quality flag
    '''
    SA = list(pd.unique(data['SA']))
    data_SA = data['SA'].to_numpy()
    idx = data.index
    SA_xco2raw_RMSE = []
    SA_xco2_RMSE = []
    SA_xco2corr_RMSE = []

    assert len(SA) > 0, 'No data to calculate variability'
    assert data_SA[0] <= data_SA[-1], 'SA needs to be sorted'
    print('get idx for SAs')
    result = collections.defaultdict(list)
    for i in tqdm(range(len(data_SA))):
        SA_i = data_SA[i]
        result[SA_i].append(i)

    for i in tqdm(range(len(SA))):
        a = SA[i]
        idx_SA = result[a]
        if len(idx_SA) > 2:
            SA_xco2raw_RMSE.append((data['xco2raw_SA_bias'].iloc[idx_SA] ** 2).mean() ** 0.5)
            SA_xco2_RMSE.append((data['xco2_SA_bias'].iloc[idx_SA] ** 2).mean() ** 0.5)
            SA_xco2corr_RMSE.append((data['xco2raw_SA_bias-ML'].iloc[idx_SA] ** 2).mean() ** 0.5)


    # plot distribution
    plt.figure(figsize=(4, 3))
    bins = np.arange(0, np.percentile(SA_xco2_RMSE, 98), 0.1)
    plt.hist(SA_xco2raw_RMSE, bins=bins, label='OCO-2 raw', histtype='step', color='b')
    plt.hist(SA_xco2_RMSE, bins=bins, label='OCO-2 B11', histtype='step', color='r')
    plt.hist(SA_xco2corr_RMSE, bins=bins, label='OCO-2 corr.', histtype='step', color='k')

    plt.title('Variability based on SA for QF=' + str(qf))
    plt.xlabel('Standard deviation [ppm]')
    plt.legend()
    plt.tight_layout()
    if save_fig:
        paths.ensure_dir_exists(path)
        save_filepath = path / (name + '_' + var_tp + '_VariabilityReduction_QF' + str(qf) + '.png')
        plt.savefig(save_filepath)
    plt.close()



def train_test_split_SA(data, test_size=0.3):
    '''make train test split based on SA

    :param data: DataFrame
    :param test_size: float, size of test set
    :return:
    '''
    print('perform train test split per SA')
    SA = list(pd.unique(data['SA']))
    iloc_all = np.arange(len(data))
    SA_train, SA_test = train_test_split(SA, test_size=test_size, random_state=1)

    idx_train = []
    idx_test = []
    # find indexes for train and test samples
    for a in SA_train:
        idx_train.append(iloc_all[data['SA'] == a])
    for a in SA_test:
        idx_test.append(iloc_all[data['SA'] == a])

    # make train test data sets
    data_train = data.iloc[np.concatenate(idx_train)]
    data_test = data.iloc[np.concatenate(idx_test)]

    return data_train, data_test


def train_test_split_Time(data, test_size=0.3):
    ''' make train test split based on Time (expects data to be ordered by Time)

    :param data: DataFrame ordered by time
    :param test_size: float, size of test set
    :return:
    '''
    print('perform train test split by time')
    n_soundings = len(data)

    data_train = data.iloc[:int(n_soundings * (1 - test_size))]
    data_test = data.iloc[int(n_soundings * (1 - test_size)):]

    return data_train, data_test


def get_importance(rf, X, name, dir: Path, save_IO=False):
    '''
    Get the importance of input features
    :param rf: RandomForestRegressor, random forest model
    :param X: pd.DataFrame, input features
    :param name: str, name
    :param dir: Path, save location
    :param save_IO: bool, save figure
    '''
    # get importance of input features
    #result = rf.feature_importances_
    result = np.array([tree.feature_importances_ for tree in rf.estimators_])
    result_mean = np.mean(result,0)
    sorted_idx = result_mean.argsort()

    if save_IO:
            # plot importances
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.boxplot(result[:,sorted_idx],vert=False, labels=X.columns[sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_xlim(0,0.5)
        plt.xscale('log')
        plt.legend()
        fig.tight_layout()
        paths.ensure_dir_exists(dir)
        plt.savefig(dir / (name + '_feature_importance.png'))
        plt.close()

    return result[:,sorted_idx]


def raster_data(Var, Lat, Lon, res=1, aggregate='mean'):
    ''' puts data on a raster (matrix) so it can be used for plotting

    :param Var: ndarray; variable we want to plot
    :param Lat: ndarray; Latitude
    :param Lon: ndarray; Longitude
    :param res: int, Resolution of raster
    :param aggregate: str, ['mean', 'count'] calc mean; count number of soundings
    :return:
    '''
    world_map = np.zeros((180 // res, 360 // res)) * np.nan

    for Lat_i in range(-90, 90, res):
        match = (Lat >= Lat_i) & (Lat < Lat_i + res)

        Var_lat = Var[match]
        Lon_lat = Lon[match]

        for Lon_i in range(-180, 180, res):
            # find values for bin
            bin = Var_lat[(Lon_lat >= Lon_i) & (Lon_lat < Lon_i + res)]
            # check if we have measurements in bin
            if len(bin) < 10:  # if there are no measurements in a grid cell
                bin = np.nan
            else:
                if aggregate == 'mean':
                    bin = np.mean(bin)  # calculate mean of values in bin
                if aggregate == 'median':
                    bin = np.median(bin)  # calculate mean of values in bin
                if aggregate == 'count':
                    bin = len(bin)  # calculate mean of values in bin
            world_map[-(Lat_i // res + 90 // res), Lon_i // res + 180 // res] = bin

    return world_map


def clean_raster_data(raster):
    """
    Clean raster data by removing NaN and infinite values
    
    :param raster: Input raster data (numpy array or similar)
    :return: Cleaned raster data with NaN and inf values replaced
    """
    raster_array = np.array(raster)
    
    # Replace infinite values with NaN
    raster_array[np.isinf(raster_array)] = np.nan
    
    # Check if we have any valid data
    if not np.isfinite(raster_array).any():
        print("Warning: All values in raster are NaN or infinite")
        return None
    
    return raster_array


def Earth_Map_Raster(raster, MIN, MAX, var_name, Title, Save=False, Save_Name: str = 'None', res=1,colormap=plt.cm.coolwarm,
                    extend = 'both'):
    ''' makes beautiful plot of data organized in a matrix representing lat lon of the globe

    :param raster: gridded data
    :param MIN: min value for plot
    :param MAX: max value for plot
    :param var_name: Name of var to be plotted
    :param Title:
    :param Save: wether to save the plot or show it
    :param Save_Name:
    :param res: resolution of map in deg
    :param colormap: what colormap to use
    :param extend: add triangles to sides of colorbar ['neither', 'both']
    :return:
    '''
    # New version with Cartopy
    limits = [-180, 180, -90, 90]
    offset = res / 2 * np.array([0, 0, 2, 2])
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    # Clean the raster data first
    raster_clean = clean_raster_data(raster)
    if raster_clean is None:
        print("Warning: No valid data in raster, skipping plot")
        return
    
    # Create masked array for invalid values
    raster_masked = np.ma.masked_invalid(np.flipud(raster_clean))
    
    # Additional check: ensure we have some valid data
    if raster_masked.count() == 0:
        print("Warning: No valid data points in raster after masking")
        return
    
    # Final check: ensure the masked array doesn't contain any problematic values
    # that could cause issues with cartopy's coordinate transformation
    if hasattr(raster_masked, 'data'):
        # Check if the underlying data contains any problematic values
        data_check = raster_masked.data
        if np.any(np.isinf(data_check)) or np.any(np.isnan(data_check)):
            # Replace any remaining problematic values with NaN
            data_check = np.where(np.isfinite(data_check), data_check, np.nan)
            raster_masked = np.ma.masked_invalid(data_check)
    
    # Ensure all coordinate values are finite for cartopy
    # This is critical for cartopy's coordinate transformation
    if not np.all(np.isfinite(limits)):
        print("Warning: Invalid coordinate limits detected, using default values")
        limits = [-180, 180, -90, 90]
    
    # Ensure the extent array is finite
    extent_array = np.array(limits) + offset
    if not np.all(np.isfinite(extent_array)):
        print("Warning: Invalid extent array detected, using default values")
        extent_array = np.array([-180, 180, -90, 90]) + offset
    
    # Handle NaN values by setting a specific color for them
    colormap.set_bad('white', alpha=0)  # Set NaN values to transparent white
    
    try:
        im = ax.imshow(raster_masked, interpolation='nearest', origin='lower',
                       extent=extent_array, cmap=colormap, vmin=MIN, vmax=MAX,
                       transform=ccrs.PlateCarree(), alpha=0.9)
    except ValueError as e:
        if "must be finite" in str(e):
            print(f"Warning: Cartopy coordinate transformation failed due to non-finite values. Using fallback method.")
            # Fallback: use a simple imshow without cartopy transformation
            # Create a simple extent without coordinate transformation
            simple_extent = [-180, 180, -90, 90]
            im = ax.imshow(raster_masked, interpolation='nearest', origin='lower',
                           extent=simple_extent, cmap=colormap, vmin=MIN, vmax=MAX, alpha=0.9)
        else:
            raise e

    ax.set_title(Title, fontsize=15, pad=10)
    plt.colorbar(im, fraction=0.066, pad=0.08, extend=extend, location='bottom', label=var_name)

    ax.coastlines()
    plt.tight_layout()
    if Save and Save_Name != 'None':
        save_path_obj = Path(Save_Name)
        paths.ensure_dir_exists(save_path_obj.parent)
        plt.savefig(save_path_obj.with_suffix('.png'), dpi=200, bbox_inches='tight')
    elif not Save:
        plt.show()
    plt.close()


def RMSE_tccon(data):
    '''
    Calculate the RMSE of the bias correction to TCCON
    :param data: pd.DataFrame, data
    :return: float, RMSE of the bias correction to TCCON
    '''
    # get soundings where we have TCCON data
    data_t = data[data['xco2tccon'] > 0]
    # compare bias correction to tccon
    diff_ML = data_t['xco2MLcorr'] - data_t['xco2tccon']
    RMSE_ML = get_RMSE(diff_ML)

    diff_B10 = data_t['xco2'] - data_t['xco2tccon']
    RMSE_B10 = get_RMSE(diff_B10)

    return RMSE_B10, RMSE_ML


def filter_TCCON(data, TCCON_names):
    ''' Filter TCCON stations based on name

    :param data: pd DataFrame, data
    :param TCCON_names: np.array or str, TCCON names to keep
    :return: pd DataFrame, data that contains only soundings that match TCCON_names
    '''
    keep = np.zeros_like(data['tccon_name'], dtype=bool)
    #if TCCON names is a single string make a list out of it
    if isinstance(TCCON_names, str):
        TCCON_names = [TCCON_names]
    for T_name in TCCON_names:
        keep[data['tccon_name'] == T_name] = True
    return data[keep]


def plot_tccon(data, TCCON_names, save_fig=False, path: Path = None, name: str = 'None', qf=None, precorrect_IO=False):
    '''

    :param data:
    :param TCCON_names: np.array or str, TCCON names to keep
    :param save_fig:
    :param path:
    :param name:
    :param qf: int, only used for plotting
    :param precorrect_IO: bool, only used for plotting
    :return:
    '''
    # get soundings where we have TCCON data
    data = data[data['xco2tccon'] > 0]

    # get testing TCCON stations

    data = filter_TCCON(data, TCCON_names)

    if len(data) <= 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan

    # compare bias correction to tccon
    diff_ML = data['xco2MLcorr'] - data['xco2tccon']
    xco2ML_std = np.std(diff_ML)
    xco2ML_median = np.median(diff_ML)
    xco2ML_RMSE = get_RMSE(diff_ML)

    print('')
    print('ML correction to tccon')
    print('STD: XCO2 corrected - XCO2 tccon: ' + str(xco2ML_std))
    print('median: XCO2 corrected - XCO2 tccon: ' + str(xco2ML_median))

    diff_B11 = data['xco2'] - data['xco2tccon']
    xco2B11_std = np.std(diff_B11)
    xco2B11_median = np.median(diff_B11)
    xco2B11_RMSE = get_RMSE(diff_B11)
    print('')
    print('B11 to tccon')
    print('STD: XCO2bc - XCO2 tccon: ' + str(xco2B11_std))
    print('median: XCO2bc - XCO2 tccon: ' + str(xco2B11_median))

    if precorrect_IO:
        diff_raw = data['xco2_raw_orig'] - data['xco2tccon']
    else:
        diff_raw = data['xco2_raw'] - data['xco2tccon']
    xco2raw_std = np.std(diff_raw)
    xco2raw_median = np.median(diff_raw)
    xco2raw_RMSE = get_RMSE(diff_raw)
    print('')
    print('Raw to tccon')
    print('STD: XCO2bc - XCO2 tccon: ' + str(xco2raw_std))
    print('median: XCO2bc - XCO2 tccon: ' + str(xco2raw_median))

    # plot distribution if we have some data

    plt.figure(figsize=(8, 4))
    bins = np.arange(np.percentile(diff_ML, 2), np.percentile(diff_ML, 98), 0.1)
    n = plt.hist(diff_ML, bins=bins, label='OCO-2 corr. - TCCON', histtype='step', color='k')
    plt.hist(diff_B11, bins=bins, label='OCO-2 B11 - TCCON', histtype='step', color='r')
    plt.hist(diff_raw, bins=bins, label='OCO-2 raw - TCCON', histtype='step', color='b')
    plt.vlines(xco2ML_median, 0, np.max(n[0]), colors='k',
               label='OCO-2 corr. = ' + str(np.round(xco2ML_median, 2)) + r'$\pm$' + str(np.round(np.std(diff_ML), 2)))
    plt.vlines(xco2B11_median, 0, np.max(n[0]), colors='r',
               label='OCO-2 B11 = ' + str(np.round(xco2B11_median, 2)) + r'$\pm$' + str(
                   np.round(np.std(diff_B11), 2)))
    plt.vlines(xco2raw_median, 0, np.max(n[0]), colors='b',
               label='OCO-2 raw = ' + str(np.round(xco2raw_median, 2)) + r'$\pm$' + str(
                   np.round(np.std(diff_raw), 2)))

    plt.title('OCO-2 - TCCON QF=' + str(qf))
    plt.xlabel('XCO2 [ppm]')
    plt.legend()
    plt.tight_layout()
    if save_fig and path is not None and name != 'None':
        paths.ensure_dir_exists(path)
        file_out = path / (name + 'Bias_vs_TCCON_QF' + str(qf) + '.png')
        plt.savefig(file_out, dpi=300)
        
    else:
        plt.show()
    plt.close()
    
    return xco2ML_std, xco2ML_median, xco2B11_std, xco2B11_median, xco2raw_std, xco2raw_median, xco2ML_RMSE, xco2B11_RMSE,xco2raw_RMSE



def plot_map(data, vars, save_fig=False, path: Path = None, name: str = 'None', pos_neg_IO = True, max=None, min=None, aggregate='mean', cmap=None):
    '''
    :param data: pd.DataFrame
    :param vars: list; vars we wish to plot
    :param save_fig: bool
    :param path: str; save path
    :param name: str; name for saving
    :param pos_neg_IO: bool; changes colorbar and min max of colorbar
    :param aggregate: str, ['mean', 'count'] calc mean; count number of soundings
    :return: image

    '''
    print('plotting maps for ' + str(vars))
    res = 2

    # make vars into a list in case we didn't pass a list
    if isinstance(vars,str):
        vars = [vars]

    for var in vars:
        # Check if the variable exists in the data
        if var not in data.columns:
            print(f"Warning: Variable '{var}' not found in data, skipping...")
            continue
            
        # Get the data for this variable
        var_data = data[var].to_numpy()
        lat_data = data['latitude'].to_numpy()
        lon_data = data['longitude'].to_numpy()
        
        # Check for valid data
        if len(var_data) == 0 or len(lat_data) == 0 or len(lon_data) == 0:
            print(f"Warning: No data available for variable '{var}', skipping...")
            continue
            
        # Check for NaN or infinite values in coordinates
        if np.any(np.isnan(lat_data)) or np.any(np.isnan(lon_data)) or np.any(np.isinf(lat_data)) or np.any(np.isinf(lon_data)):
            print(f"Warning: Invalid coordinates found for variable '{var}', cleaning data...")
            # Remove rows with invalid coordinates
            valid_mask = np.isfinite(lat_data) & np.isfinite(lon_data)
            if not np.any(valid_mask):
                print(f"Warning: No valid coordinates for variable '{var}', skipping...")
                continue
            var_data = var_data[valid_mask]
            lat_data = lat_data[valid_mask]
            lon_data = lon_data[valid_mask]
        
        raster = raster_data(var_data, lat_data, lon_data, res=res, aggregate=aggregate)
        if pos_neg_IO:
            if max == None:
                MAX = np.abs(np.nanpercentile(raster, 95))
                MIN = np.abs(np.nanpercentile(raster, 5))
                MAXX = np.max([MAX, MIN])
                MIN = -MAXX
                MAX = MAXX
            else:
                MIN = min
                MAX = max
            if cmap == None:
                colormap = plt.cm.coolwarm
            else:
                colormap = cmap
            extend = 'both'
        else:
            if max == None:
                MAX = np.nanpercentile(raster, 95)
            else:
                MAX = max
            if min == None:
                MIN = np.nanpercentile(raster,5)
            else:
                MIN = min
            if cmap == None:
                colormap = plt.cm.get_cmap('OrRd')
            else:
                colormap = cmap
            extend = 'max'
        var_name = var
        # plot the map
        Earth_Map_Raster(raster, MIN, MAX, var_name, name, res=res, Save=save_fig,
                         Save_Name= Path(path) / (name + var), colormap=colormap, extend = extend)



def plot_qf_map(data_qf,data_mlqf, vars,year = '2021', save_fig=False, path: Path =None, name: str ='None',diff = True):
    '''plot spatial histograms
    :param data_qf: pd.DataFrame for B11 QF
    :param data_mlqf: pd.DataFrame for ML QF
    :param vars: list; vars we wish to plot
    :param save_fig: bool
    :param path: str; save path
    :param name: str; name for saving
    :param diff: bool; changes colorbar and min max of colorbar
    :return: image
    '''
    if not diff:
        result, xedges, yedges = np.histogram2d(data_mlqf['longitude'],data_mlqf['latitude'],bins=[90, 45])
        #result = result.clip(1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.Robinson())
        ax.coastlines()
        ax.gridlines()
        im = ax.imshow(result.T, interpolation='nearest', origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap = 'Blues',vmin = 100, vmax = 2000, transform=ccrs.PlateCarree(), alpha = 0.7)

        ax.set_title("MLQF = 0 " + str(vars) + " " + year)
        plt.colorbar(im, fraction=0.066, pad=0.04, extend='max', location = 'bottom', label='# soundings')
        if save_fig and path is not None:
            paths.ensure_dir_exists(path)
            plt.savefig(path / ("Percent_pass_" + str(vars) + "_"+ year+ ".jpg"), dpi = 300)
        else:
            plt.show()
        plt.close()
    else:
        result1, xedges, yedges = np.histogram2d(data_mlqf['longitude'],data_mlqf['latitude'],bins=[90, 45])
        result2, _, _ = np.histogram2d(data_qf['longitude'],data_qf['latitude'],bins=[xedges, yedges])
        result = (result1 - result2) #/result1 * 100
        #result = result.clip(1)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.Robinson())
        ax.coastlines()
        ax.gridlines()
        im = ax.imshow(result.T, interpolation='nearest', origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap = 'PuOr',vmin = -250, vmax = 250, transform=ccrs.PlateCarree(), alpha = 0.7)

        ax.set_title("MLQF = 0 " + str(vars) + " " + year)
        plt.colorbar(im, fraction=0.066, pad=0.04, extend='both', location = 'bottom', label='Diff in % passing')
        if save_fig and path is not None:
            paths.ensure_dir_exists(path)
            plt.savefig(path / ("Difference_pass_" + str(vars) + "_"+ year+ ".jpg"), dpi = 300)
        else:
            plt.show()
        plt.close()
            
            
def hex_plot(data,name: str, path: Path, save_fig = False,var1 = ['h2o_ratio'],var2 = ['dpfrac'], bias = 'xco2raw_SA_bias'):
    '''
    Plot the hexbin plot of the bias vs the input features
    :param data: pd.DataFrame, data
    :param name: str, name
    :param path: Path, save location
    :param save_fig: bool, save figure
    '''
    for idx, var in enumerate(var1):
        x = data[var]
        y = data[var2[idx]]
        z = data[bias]


        f, ax = plt.subplots(figsize=(5, 4))
        plt.hexbin(x=x,y=y,C=z, cmap = "RdBu", alpha = 1, gridsize = 20)
        plt.axvline(x=0.75, linestyle='--', color='black', lw=2) # TODO: add tuple as input
        plt.axvline(x=1.07, linestyle='--', color='black', lw=2)
        plt.axhline(y=-3.5, linestyle='--', color='black', lw=2)
        plt.axhline(y=3.0, linestyle='--', color='black', lw=2)

        plt.clim(-1,1)
        plt.colorbar(extend='both',  label='\u03B4' + 'XCO2 [ppm]')
        plt.xlabel(var)
        plt.ylabel(var2[idx])
        plt.tight_layout()
        if save_fig:
            paths.ensure_dir_exists(path)
            plt.savefig(path / (name + "_" + var + "_vs_" + var2[idx] + "_hex.png"), dpi = 300) # More descriptive name
            plt.close()
        else:
            plt.show()
            
def plot_decision_surface(M, data_test,y_test_c,save_fig = True, file_path: Path = None):
    '''
    Plot the decision surface of the model
    :param M: RandomForestRegressor, random forest model
    :param data_test: pd.DataFrame, test data
    :param y_test_c: pd.Series, test data
    :param save_fig: bool, save figure
    :param file_path: Path, save location
    '''
    feature_names = data_test.columns
    # Create the scatter plots
    # fig, axs = plt.subplots(npredictors+1, npredictors+1, figsize=(15, 15))
    # fig.subplots_adjust(wspace=.35)
    for f1 in range(feature_names+1):
        for f2 in range(feature_names+1): 
            if f1 == f2:
                continue;
            else:
                x1 = data_test.loc[:,f1]
                x2 = data_test.loc[:,f2]
                X = pd.concat([x1,x2], axis = 1)
                disp = DecisionBoundaryDisplay.from_estimator(
                     M, X, response_method="predict",
                     alpha=0.5, xlabel = feature_names[f1], ylabel = feature_names[f2]
                )
                disp.ax_.scatter(data_test[:, 0], data_test[:, 1], c=y_test_c, edgecolor="k")
                if save_fig:
                    disp.save_fig(file_path, dpi = 200)
                else:
                    plt.show()
                plt.close()
        
def dist(lat1, lat2, lon1, lon2):
    '''distance calculation between points given degree
    :param lat1:
    :param lat2:
    :param lon1:
    :param lon2:
    :return: distance in km
    '''

    # transforms deg to rad
    lat1 = lat1*np.pi / 180.
    lat2 = lat2 * np.pi / 180.
    lon1 = lon1 * np.pi / 180.
    lon2 = lon2 * np.pi / 180.
    #calculate distance in km between two points in spherical coordinates


    d = np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2))*6371
    return d

def bias_correct(path: Path, data, vars_to_correct, uq = False, proxy_name = 'TCCON'):
    ''' perform bias correction with a trained model

    :param vars_to_correct: lst, variables we want to correct
    :param path: Path, path to model directory (which contains trained_model.joblib and normalization_params.json)
    :param data: dataframe, processed soundings
    :return: dataframe, bias corrected soundings
    '''
    assert proxy_name in ['TCCON','SA']
    
    # Load the trained model
    model_file = path / 'trained_model.joblib'
    if not model_file.exists():
        raise FileNotFoundError(f"Trained model file not found: {model_file}")
    M = joblib.load(model_file)

    # Load normalization parameters and model metadata
    params_file = path / 'normalization_params.json'
    if not params_file.exists():
        # This should not happen anymore as 11_train_bias_correction_model.py always creates it.
        raise FileNotFoundError(f"Normalization parameters file not found: {params_file}. This file should always be present.")

    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # X_mean and X_std might be present but not used by RF for input normalization during predict, 
    # but are used for y_mean/y_std if the model output was normalized (which it is not for RF here)
    X_mean = pd.Series(params['X_mean']) 
    X_std = pd.Series(params['X_std'])
    y_mean = params['y_mean'] 
    y_std = params['y_std']
    features = params['features']
    model_type = params['model_type'] 

    # make input output pair
    X = data[features]
    # calculate bias
    print('calculating bias correction')
    if model_type == 'RF':

        bias, bias_std = make_prediction(M, X, model_type, UQ = uq)
    elif model_type in ['GPR', 'NN', 'Ridge', 'XGB', 'BayesianRidge']:
        # These models require X_mean and X_std for input normalization
        if X_mean is None or X_std is None: # Should not happen if JSON is always created correctly
             raise ValueError("Normalization parameters (X_mean, X_std) are missing or not pd.Series for a model type that requires them.")
        # normalize data to 0 mean unit standard deviation
        X_norm = (X - X_mean) / X_std
        bias, bias_std = make_prediction(M, X_norm, model_type, y_mean=y_mean, y_std=y_std)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    #remove bias
    for v in vars_to_correct:
        data[v] = data[v] - bias

    if uq == True:
        # data[proxy_name +'_bias_uq'] = bias_std # for testing purposes
        data['bias_correction_uncert'] = bias_std

    return data


def construct_filter(data, path_tc_lnd: Path, path_tc_ocn: Path, path_sa_lnd: Path, path_sa_ocn: Path, abstention_threshold_lnd = 1.3, abstention_threshold_ocn = 1.05):
    ''' Construct the ternary filter flag for B112

    :param path_tc_lnd: Path, path to tccon land model directory
    :param path_tc_ocn: Path, path to tccon ocean model directory
    :param path_sa_lnd: Path, path to small area land model directory
    :param path_sa_ocn: Path, path to small area ocean model directory
    :param abstention_threshold_lnd: float, threshold for UQ based filtering
    :param abstention_threshold_ocn: float, threshold for UQ based filtering
    :return: data frame with quality flag 'xco2_quality_flag_b112'
    '''

    data['aod_diff'] = data['aod_total'] - data['aod_total_apriori']
    
    # Load TCCON and Small Area Filter models

    M_tc_lnd = joblib.load(path_tc_lnd.with_suffix('.joblib'))
    features_tc_lnd = M_tc_lnd.feature_names_in_

    M_tc_ocn = joblib.load(path_tc_ocn.with_suffix('.joblib'))
    features_tc_ocn = M_tc_ocn.feature_names_in_

    # Small Area Filter models
    M_sa_lnd = joblib.load(path_sa_lnd.with_suffix('.joblib'))
    features_sa_lnd = M_sa_lnd.feature_names_in_

    M_sa_ocn = joblib.load(path_sa_ocn.with_suffix('.joblib'))
    features_sa_ocn = M_sa_ocn.feature_names_in_


    # get predictions
    land = data[data['land_water_indicator'] == 0]
    ocean = data[data['land_water_indicator'] != 0]

    # check that we have land and ocean soundings
    if len(land) > 0:
        # handle NaNs
        # Find rows with NaN values in pandas DataFrame
        nan_rows_land = pd.isna(land).any(axis=1)
        # If any NaN values are detected, print a warning
        if np.any(nan_rows_land):
            print("Warning: NaN values detected. Setting prediction to 2 for these samples.")
            land['TCCON_flag'] = 2
            land['SA_flag'] = 2

            # Find rows with NaN values
            land['TCCON_flag'][~nan_rows_land] = M_tc_lnd.predict(land[features_tc_lnd][~nan_rows_land])
            land['SA_flag'][~nan_rows_land] = M_sa_lnd.predict(land[features_sa_lnd][~nan_rows_land])
        else:
            land['TCCON_flag'] = M_tc_lnd.predict(land[features_tc_lnd])
            land['SA_flag'] = M_sa_lnd.predict(land[features_sa_lnd])

        land['xco2_quality_flag_ML'] = 2
        land.loc[(land['TCCON_flag'] == 0) | (land['SA_flag'] == 0), 'xco2_quality_flag_ML'] = 1
        land.loc[(land['TCCON_flag'] == 0) & (land['SA_flag'] == 0) & (np.sqrt((land['bias_correction_uncert'] ** 2) - (land['xco2_uncertainty'] ** 2)) <= abstention_threshold_lnd), 'xco2_quality_flag_ML'] = 0

    if len(ocean) > 0:
        # handle NaNs
        # Find rows with NaN values
        nan_rows_ocean = pd.isna(ocean).any(axis=1)
        # If any NaN values are detected, print a warning
        if np.any(nan_rows_ocean):
            print("Warning: NaN values detected. Setting prediction to 2 for these samples.")
            ocean['TCCON_flag'] = 2
            ocean['SA_flag'] = 2

            # Find rows with NaN values
            ocean['TCCON_flag'][~nan_rows_ocean] = M_tc_ocn.predict(ocean[features_tc_ocn][~nan_rows_ocean])
            ocean['SA_flag'][~nan_rows_ocean] = M_sa_ocn.predict(ocean[features_sa_ocn][~nan_rows_ocean])
        else:
            ocean['TCCON_flag'] = M_tc_ocn.predict(ocean[features_tc_ocn])
            ocean['SA_flag'] = M_sa_ocn.predict(ocean[features_sa_ocn])
        
        ocean['xco2_quality_flag_ML'] = 2
        ocean.loc[(ocean['TCCON_flag'] == 0) | (ocean['SA_flag'] == 0), 'xco2_quality_flag_ML'] = 1
        ocean.loc[(ocean['TCCON_flag'] == 0) & (ocean['SA_flag'] == 0) & (np.sqrt((ocean['bias_correction_uncert'] ** 2) - (ocean['xco2_uncertainty'] ** 2)) <= abstention_threshold_ocn), 'xco2_quality_flag_ML'] = 0


    # if we have land and ocean soundings concatenate them
    if len(land) > 0 and len(ocean) > 0:
        data = pd.concat([land, ocean])
    elif len(land) > 0:
        data = land
    elif len(ocean) > 0:
        data = ocean
    else:
        # throw exception if we don't have any soundings
        raise Exception('No soundings to filter')
    
    # sort data by sounding id 
    data = data.sort_values(by=['sounding_id'])

    # drop the temp flags, and additional filter vars not in L2 Lite from the dataframe
    data = data.drop(['TCCON_flag', 'SA_flag', 
                      'aod_diff', ], axis=1)
                  

    return data

def calc_SA_bias(XCO2, SA):
    '''
    :param XCO2: XCO2
    :param SA: SA id
    :return: XCO2 - median of XCO2 for a given SA id
    '''
    print('recalculating SA bias')
    SA_unique = np.unique(SA)

    ## faster way (SA needs to be sorted)
    assert SA[0] <= SA[-1], 'SA needs to be sorted'

    result = collections.defaultdict(list)
    for val, idx in zip(XCO2.ravel(), SA.ravel()):
        result[idx].append(val)
    xco2raw_SA_biases = []
    for idx in SA_unique:
        if len(result[idx]) > 10: # make sure we have enough soundings to calc SA bias
            xco2raw_SA_biases.append(result[idx] - np.median(result[idx]))
        else:
            xco2raw_SA_biases.append(np.array(result[idx]) * np.nan)
    xco2raw_SA_bias = np.concatenate(xco2raw_SA_biases)

    return xco2raw_SA_bias

def calc_SA_bias_clean(XCO2, SA, qf):
    '''Calculate SA bias but only allow high quality soundings in median to calc true XCO2
    :param XCO2: XCO2
    :param SA: SA id
    :return: XCO2 - median of XCO2 for a given SA id
    '''
    print('recalculating SA bias')
    SA_unique = np.unique(SA)

    ## faster way (SA needs to be sorted)
    assert SA[0] <= SA[-1], 'SA needs to be sorted'
    # find XCO2 values that belong to one SA
    SA_dict_XCO2 = collections.defaultdict(list)
    for val, idx in zip(XCO2.ravel(), SA.ravel()):
        SA_dict_XCO2[idx].append(val)
    # find quality values that belong to one SA
    SA_dict_qf = collections.defaultdict(list)
    for val, idx in zip(qf.ravel(), SA.ravel()):
        SA_dict_qf[idx].append(val)

    xco2raw_SA_biases = []
    for idx in SA_unique:
        # extract XCO2 for a given SA
        XCO2_of_SA = np.array(SA_dict_XCO2[idx])
        # extract quality flag from SA
        qf_of_SA = np.array(SA_dict_qf[idx])
        # check that we have at least one high quality sounding to calcultate median
        if np.count_nonzero(qf_of_SA) >= len(qf_of_SA)-5:
            xco2raw_SA_biases.append(XCO2_of_SA - np.nan)
        else:
            # calculate truth (median) only from high quality data (with QF=0)
            xco2raw_SA_biases.append(XCO2_of_SA - np.median(XCO2_of_SA[qf_of_SA == 0]))
    xco2raw_SA_bias = np.concatenate(xco2raw_SA_biases)

    return xco2raw_SA_bias



def get_season(data):
    '''
    Get the season of the data
    :param data: pd.DataFrame, data
    :return: pd.DataFrame, data with season
    '''
    Date = data['sounding_id'].to_numpy()
    Month_List = []
    Year_List = []
    for d in Date:
        d = str(d)
        Year = int(d[0:4])
        Month = list(map(int, d[4:6]))
        if Month[0] == 0:
            Month = Month[1]
        else:
            Month = Month[1] + 10
        Month_List.append(Month)
        Year_List.append(Year)

    Months = np.stack(Month_List)
    Years = np.stack(Year_List)

    data.loc[(Months >= 3) & (Months <= 5), 'season'] = 'MAM'  # Mar,Apr,May
    data.loc[(Months >= 6) & (Months <= 8), 'season'] = 'JJA'  # Jun,Jul,Aug
    data.loc[(Months >= 9) & (Months <= 11), 'season'] = 'SON'  # Sept,Oct,Nov
    data.loc[(Months == 12) | (Months == 1) | (Months == 2), 'season'] = 'DJF'  # Dec, Jan, Feb

    data.loc[:,'Month'] = Months
    data.loc[:,'Year'] = Years

    return data


def weight_TCCON(data, features):
    '''
    Weight based on number of samples per TCCON station
    :param data: pd.DataFrame, data
    :param features: list, features
    :return: pd.DataFrame, data with weights
    '''
    # weight based on number of samples per TCCON station
    TCCON_names = np.unique(data['tccon_name'])
    weights = np.zeros(len(TCCON_names))
    for i in range(len(TCCON_names)):
        weights[i] = 1 / len(data[data['tccon_name'] == TCCON_names[i]])
    weights = weights / np.sum(weights)
    data['weights'] = np.zeros(len(data))
    for i in range(len(TCCON_names)):
        data.loc[data['tccon_name'] == TCCON_names[i], 'weights'] = weights[i]
    return data

   
    return data


def weight_coast(data, multiplier=100):
    '''
    Weight land ocean crossings more
    :param data: pd.DataFrame, data
    :param multiplier: float, multiplier
    :return: pd.DataFrame, data with weights
    '''
    # weight land ocean crossings more
    weights = np.ones(len(data))
    weights[data['coast'] == 1] = multiplier
    # make sure weights sum to 1
    weights = weights / np.sum(weights)
    data['weights'] = weights

    return data


def feature_selector(feature_select):
    ''' select a set of features

    :param feature_select: name of Feature set e.g. 'SA_bias_LndND'
    :param application: str, SA_bias: small areas; TCCON_bias: tccon; SA_filt: Filtering based on small areas
    :return: list of features
    '''
    # bias correction *********************************************************
    if feature_select == 'TCCON_bias_all':
        features = ['co2_grad_del', 'dpfrac', 'dp_o2a','dp_sco2', 'aod_strataer', 'water_height', 'footprint', 'aod_water','aod_ice','albedo_o2a', 'airmass', 'dws']#, 'albedo_slope_sco2', 'dp_o2a']
        feature_n = 'all_'
    if feature_select == 'SA_bias_all':
        features = ['color_slice_noise_ratio_o2a', 'albedo_o2a', 'dpfrac', 'albedo_slope_sco2', 'h_continuum_o2a', 'h2o_scale',
                    'co2_grad_del', 'aod_strataer', 'water_height', 'footprint', 'aod_water','aod_ice']
        feature_n = 'all_'
    if feature_select == 'SA_bias_LndNDGL':
        features = ['co2_grad_del','dpfrac', 'footprint', 'dws']
        feature_n = 'lnd_'
    if feature_select == 'SA_bias_LndND':
        features = ['co2_grad_del','dpfrac', 'albedo_o2a','rms_rel_o2a','albedo_slope_wco2','aod_strataer', 'water_height',
                     'dp_abp', 'color_slice_noise_ratio_o2a', 'h_continuum_sco2',  'rms_rel_sco2', 'footprint', 'dws']
        feature_n = 'lnd_'
    if feature_select == 'SA_bias_LndGL':
        features = ['co2_grad_del','dpfrac', 'rms_rel_o2a','aod_strataer', 'airmass','footprint',
                     'dp_o2a', 'aod_ice','dws','h_continuum_sco2']
        feature_n = 'lnd_'
    if feature_select == 'SA_bias_Sea':
        features = ['footprint', 'co2_grad_del', 'color_slice_noise_ratio_o2a','albedo_o2a']
        feature_n = 'sea_'
    if feature_select == 'TCCON_bias_Lnd':
        features = ['co2_grad_del','dpfrac', 'aod_strataer', 'albedo_sco2','dust_height','water_height',  'h2o_ratio_bc',  'aod_sulfate', 'aod_ice', 'ice_height', 'dws']
        feature_n = 'lnd_'
    if feature_select == 'TCCON_bias_Sea':
        features = ['albedo_o2a', 'albedo_slope_sco2', 'co2_grad_del', 'dp','aod_dust','h_continuum_o2a', 'water_height', 'ice_height', 'sensor_zenith_angle']
        feature_n = 'sea_'
    # filtering **************************************************************
    if feature_select == 'SA_filt_Lnd':
        features = ['co2_grad_del','dpfrac','h_continuum_wco2','h_continuum_sco2','h_continuum_o2a', 'max_declocking_sco2','chi2_o2a','rms_rel_wco2','ice_height','water_height','co2_ratio','max_declocking_wco2']
        feature_n = 'lnd_'
        
    if feature_select == 'SA_filt_Sea':
        features = ['rms_rel_o2a', 'dp_sco2', 'co2_grad_del', 'dp','dust_height', 'deltaT','aod_ice','co2_grad_del''aod_strataer','h_continuum_sco2','aod_ice','ice_height','h_continuum_o2a','aod_sulfate','co2_ratio','albedo_sco2', 'rms_rel_wco2',
                   'rms_rel_sco2','snr_sco2','h2o_ratio']
        feature_n = 'sea_'
        
    if feature_select == 'SA_filt_all':
        features = ['co2_grad_del','dpfrac','aod_strataer','dp_o2a','h_continuum_sco2','aod_ice','ice_height','h_continuum_o2a','aod_sulfate','co2_ratio','albedo_sco2', 'rms_rel_wco2',
                   'rms_rel_sco2','footprint','solar_zenith_angle']
        feature_n = 'all_'
        
    if feature_select == 'TCCON_filt_Lnd':
        features = ['co2_grad_del','dpfrac','h_continuum_wco2','h_continuum_sco2','h_continuum_o2a', 'max_declocking_sco2','chi2_o2a','rms_rel_wco2','ice_height','water_height','co2_ratio','max_declocking_wco2']
        feature_n = 'lnd_'
        
    if feature_select == 'TCCON_filt_Sea':
        features = ['co2_grad_del','dpfrac','h_continuum_wco2','h_continuum_sco2','h_continuum_o2a', 'max_declocking_sco2','chi2_o2a','rms_rel_wco2','ice_height','water_height','co2_ratio','max_declocking_wco2']
        feature_n = 'sea_'
        
    if feature_select == 'TCCON_filt_all':
        features = ['co2_grad_del','dpfrac','h_continuum_wco2','h_continuum_sco2','h_continuum_o2a', 'max_declocking_sco2','chi2_o2a','rms_rel_wco2','ice_height','water_height','co2_ratio','max_declocking_wco2']
        feature_n = 'all_'

    return features, feature_n

def load_and_concat_years(start_year, end_year, mode, hold_out_year=None, **kwargs):
    """Loads data for a range of years, concatenates them, optionally holding out one year.

    Args:
        start_year (int): The first year to load.
        end_year (int): The last year to load (inclusive).
        mode (str): The mode argument for load_data (e.g., 'LndNDGL', 'SeaGL').
        hold_out_year (int, optional): A year to exclude from loading. Defaults to None.
        **kwargs: Additional keyword arguments to pass directly to the load_data function
                  (e.g., qf, preload_IO, TCCON, remove_inland_water).

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all loaded years.
    """
    data_list = []
    print(f"Loading data from {start_year} to {end_year}, holding out {hold_out_year or 'None'}...")
    for year in range(start_year, end_year + 1):
        if year == hold_out_year:
            print(f"  Skipping hold-out year: {year}")
            continue
        # Pass mode and other specific args captured by kwargs to load_data
        yearly_data = load_data(year, mode=mode, **kwargs)
        data_list.append(yearly_data)

    concatenated_data = pd.concat(data_list, ignore_index=True)
    return concatenated_data

def plot_histogram(data, column, save_fig=False, path: Path = None, name: str = 'None', bins=None, xlabel=None, title=None, qf_col='xco2_quality_flag', ylabel='Count'):
    """
    Plot overlaid histograms of a difference column (e.g., 'ML-B11') for qf=0 (blue), qf=1 (red), and qf=0+1 (black).
    Args:
        data (pd.DataFrame): The data to plot from.
        column (str): The column name to plot (e.g., 'ML-B11').
        save_fig (bool): Whether to save the figure.
        path (Path or None): Where to save the figure.
        name (str): Name for saving the figure.
        bins (array-like or int, optional): Bins for the histogram. If None, auto-calculated.
        xlabel (str, optional): X-axis label. If None, uses column name.
        title (str, optional): Plot title. If None, auto-generated.
        qf_col (str): The column name for quality flag (default 'xco2_quality_flag').
    """
    # Prepare data for each qf
    diff_0 = data.loc[data[qf_col] == 0, column].dropna()
    diff_1 = data.loc[data[qf_col] == 1, column].dropna()
    diff_01 = data[column].dropna()

    # If all are empty, skip
    if len(diff_0) == 0 and len(diff_1) == 0 and len(diff_01) == 0:
        print(f"No data to plot for {column}")
        return

    # Bins: use all data for bin range
    all_diff = pd.concat([diff_0, diff_1, diff_01])
    if bins is None:
        bins = np.linspace(np.percentile(all_diff, 2), np.percentile(all_diff, 98), 100)
    if xlabel is None:
        xlabel = column
    if title is None:
        title = f"{column} histogram by QF"

    plt.figure(figsize=(5, 3))

    # QF=0
    if len(diff_0) > 0:
        n0 = plt.hist(diff_0, bins=bins, label=f"QF=0: {np.round(np.median(diff_0),2)}$\\pm${np.round(np.std(diff_0),2)}", histtype='step', color='blue')
        median0 = np.median(diff_0)
        std0 = np.std(diff_0)
        # plt.vlines(median0, 0, np.max(n0[0]), colors='blue', linestyles='dashed', label=None)
    # QF=1
    if len(diff_1) > 0:
        n1 = plt.hist(diff_1, bins=bins, label=f"QF=1: {np.round(np.median(diff_1),2)}$\\pm${np.round(np.std(diff_1),2)}", histtype='step', color='red')
        median1 = np.median(diff_1)
        std1 = np.std(diff_1)
        # plt.vlines(median1, 0, np.max(n1[0]), colors='red', linestyles='dashed', label=None)
    # QF=0+1 (all)
    if len(diff_01) > 0:
        n01 = plt.hist(diff_01, bins=bins, label=f"QF=0+1: {np.round(np.median(diff_01),2)}$\\pm${np.round(np.std(diff_01),2)}", histtype='step', color='black')
        median01 = np.median(diff_01)
        std01 = np.std(diff_01)
        # plt.vlines(median01, 0, np.max(n01[0]), colors='black', linestyles='dashed', label=None)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save_fig and path is not None and name != 'None':
        paths.ensure_dir_exists(path)
        file_out = path / (f"{name}_{column}_hist_byQF.png")
        plt.savefig(file_out, dpi=300)
    else:
        plt.show()
    plt.close()
    print('Histogram saved to', file_out)