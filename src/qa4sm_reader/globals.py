# -*- coding: utf-8 -*-

"""
Settings and global variables used in the reading and plotting procedures
"""
# todo: reduce dependency on globals (e.g flexible if new datasets/versions are added)
import cartopy.crs as ccrs

# === plot defaults ===
matplotlib_ppi = 72  # Don't change this, it's a matplotlib convention.
index_names = ['lat', 'lon']  # Names used for 'lattitude' and 'longitude' coordinate.
time_name = 'time' # not used at the moment, dropped on load
period_name = 'period' # not used at the moment, dropped on load

dpi = 100  # Resolution in which plots are going to be rendered.
title_pad = 12.0  # Padding below the title in points. default padding is matplotlib.rcParams['axes.titlepad'] = 6.0
data_crs = ccrs.PlateCarree()  # Default map projection. use one of

# === map plot defaults ===
scattered_datasets = ['ISMN']  # dataset names which require scatterplots (values is scattered in lat/lon)
map_figsize = [11.32, 6.10]  # size of the output figure in inches.
naturalearth_resolution = '110m'  # One of '10m', '50m' and '110m'. Finer resolution slows down plotting. see https://www.naturalearthdata.com/
crs = ccrs.PlateCarree()  # projection. Must be a class from cartopy.crs. Note, that plotting labels does not work for most projections.
markersize = 4  # diameter of Marker in points.
map_pad = 0.15  # padding relative to map height.
grid_intervals = [2, 5, 10, 30]  # grid spacing in degree to choose from (plotter will try to make 5 gridlines in the smaller dimension)
max_title_len = 8 * map_figsize[0]  # maximum length of plot title in chars. if longer, it will be broken in multiple lines.

# === boxplot_basic defaults ===
boxplot_printnumbers = True  # Print 'median', 'nObs', 'stdDev' to the boxplot_basic.
boxplot_height = 6
boxplot_width = 2 # times (n+1), where n is the number of boxes.
boxplot_title_len = 8 * boxplot_width  # times the number of boxes. maximum length of plot title in chars.

# === watermark defaults ===
watermark = u'made with QA4SM (qa4sm.eu)'  # Watermark string
watermark_pos = 'bottom'  # Default position ('top' or 'bottom' or None)
watermark_fontsize = 8  # fontsize in points (matplotlib uses 72ppi)
watermark_pad = 5  # padding above/below watermark in points (matplotlib uses 72ppi)

# === filename template ===
ds_fn_templ = "{i}-{ds}.{var}"
ds_fn_sep = "_with_"

# === colormaps used for plotting metrics ===
# Colormaps can be set for classes of similar metrics or individually for metrics.
# Any colormap name can be used, that works with matplotlib.pyplot.cm.get_cmap('colormap')
# more on colormaps: https://matplotlib.org/users/colormaps.html | https://morphocode.com/the-use-of-color-in-maps/
# colorcet: http://colorcet.pyviz.org/user_guide/Continuous.html

import colorcet
import matplotlib.pyplot as plt
_cclasses = {
    'div_better': plt.cm.get_cmap('RdYlBu'),  # diverging: 1 good, 0 special, -1 bad (pearson's R, spearman's rho')
    'div_neutr': plt.cm.get_cmap('RdYlGn'),  # diverging: zero good, +/- neutral: (bias)
    'seq_worse': colorcet.cm['CET_L4_r'], #'cet_CET_L4_r',  # sequential: increasing value bad (p_R, p_rho, rmsd, ubRMSD, RSS):
    'seq_better': colorcet.cm['CET_L4'], #'cet_CET_L4'  # sequential: increasing value good (n_obs)
}

# 0=common metrics, 2=paired metrics (2 datasets), 3=triple metrics (TC, 3 datasets)
metric_groups = {
    0: ['n_obs'],
    2: ['R', 'p_R', 'rho','p_rho', 'RMSD', 'BIAS',
        'urmsd', 'mse', 'mse_corr', 'mse_bias', 'mse_var',
        'RSS', 'tau', 'p_tau'
        ],
    3: ['snr', 'err_std', 'beta']
}


# === variable template ===
# how the metric is separated from the rest
var_name_metric_sep = {
    0: "{metric}",
    2: "{metric}_between_",
    3: "{metric}_{mds_id:d}-{mds}_between_"
}
var_name_CI = {
    0: "{metric}_ci_{bound}_between_",
    2: "{metric}_ci_{bound}_between_",
    3: "{metric}_ci_{bound}_{mds_id:d}-{mds}_between_"
}
# how two datasets are separated, ids must be marked as numbers with :d!
var_name_ds_sep = {
    0: None, 2: "{ref_id:d}-{ref_ds}_and_{sat_id0:d}-{sat_ds0}",
    3: "{ref_id:d}-{ref_ds}_and_{sat_id0:d}-{sat_ds0}_and_{sat_id1:d}-{sat_ds1}"
}

# === metadata tempplates ===
_ref_ds_attr = 'val_ref' # global meta values variable that links to the reference dc
_ds_short_name_attr = 'val_dc_dataset{:d}' # attribute convention for other datasets
_ds_pretty_name_attr = 'val_dc_dataset_pretty_name{:d}' # attribute convention for other datasets
_version_short_name_attr = 'val_dc_version{:d}' # attribute convention for other datasets
_version_pretty_name_attr = 'val_dc_version_pretty_name{:d}' # attribute convention for other datasets

# format should have (metric, ds, ref, other ds)
_variable_pretty_name = {
    0: "{}", 2: "{} of {} \n with {} as reference",
    3: "{} of {} \n against {}, {}"
}

_colormaps = {  # from /qa4sm/validator/validation/graphics.py
    'R': _cclasses['div_better'],
    'p_R': _cclasses['seq_worse'],
    'rho': _cclasses['div_better'],
    'p_rho': _cclasses['seq_worse'],
    'RMSD': _cclasses['seq_worse'],
    'BIAS': _cclasses['div_neutr'],
    'n_obs': _cclasses['seq_better'],
    'urmsd': _cclasses['seq_worse'],
    'mse': _cclasses['seq_worse'],
    'mse_corr': _cclasses['seq_worse'],
    'mse_bias': _cclasses['seq_worse'],
    'mse_var': _cclasses['seq_worse'],
    'RSS': _cclasses['seq_worse'],
    'tau':_cclasses['div_better'],
    'p_tau': _cclasses['seq_worse'],
    'snr': _cclasses['div_better'],
    'err_std': _cclasses['div_neutr'],
    'beta': _cclasses['div_neutr'],
}
# check if every metric has a colormap
for group in metric_groups.keys():
    assert all([m in _colormaps.keys() for m in metric_groups[group]])

# Value ranges of metrics, either absolute values, or a quantile between 0 and 1
_metric_value_ranges = {  # from /qa4sm/validator/validation/graphics.py
    'R': [-1, 1],
    'p_R': [0, 1],  # probability that observed corellation is statistical fluctuation
    'rho': [-1, 1],
    'p_rho': [0, 1],
    'tau': [-1, 1],
    'p_tau': [0, 1],
    'RMSD': [0, None],
    'BIAS': [None, None],
    'n_obs': [0, None],
    'urmsd': [0, None],
    'RSS': [0, None],
    'mse': [0, None],
    'mse_corr': [0, None],
    'mse_bias': [0, None],
    'mse_var': [0, None],
    'snr': [None, None],
    'err_std': [None, None],
    'beta': [None, None],
}

# Value ranges for differences between metrics
_diff_value_ranges = {  # from /qa4sm/validator/validation/graphics.py
    'R': [-2, 2],
    'p_R': [-1, 1],  # probability that observed corellation is statistical fluctuation
    'rho': [-2, 2],
    'p_rho': [-1, 1],
    'tau': [-2, 2],
    'p_tau': [-1, 1],
    'RMSD': [None, None],
    'BIAS': [None, None],
    'n_obs': [None, None],
    'urmsd': [None, None],
    'RSS': [None, None],
    'mse': [None, None],  # mse only positive (https://en.wikipedia.org/wiki/Mean_squared_error)
    'mse_corr': [None, None],  # mse_corr only positive
    'mse_bias': [None, None],  # mse_bias only positive
    'mse_var': [None, None],  # mse_var only positive
    'snr': [None, None],  # mse_var only positive
    'err_std': [None, None],  # mse_var only positive
    'beta': [None, None],  # mse_var only positive
}

# check if every metric has a colormap
for group in metric_groups.keys():
    assert all([m in _colormaps.keys() for m in metric_groups[group]])

# label format for all metrics
_metric_description = {  # from /qa4sm/validator/validation/graphics.py
    'R': '',
    'p_R': '',
    'rho': '',
    'p_rho': '',
    'tau': '',
    'p_tau': '',
    'RMSD': r' in ${}$',
    'BIAS': r' in ${}$',
    'n_obs': '',
    'urmsd': r' in ${}$',
    'RSS': r' in $({})^2$',
    'mse': r' in $({})^2$',
    'mse_corr': r' in $({})^2$',
    'mse_bias': r' in $({})^2$',
    'mse_var': r' in $({})^2$',
    'snr': r' in $(dB)$',
    'err_std': r' in $({})$',
    'beta': r' in $({})$',
}

# units for all datasets
_metric_units = {  # from /qa4sm/validator/validation/graphics.py
    'ISMN': r'm³/m³',
    'C3S': r'm³/m³',
    'GLDAS': r'm³/m³',
    'ASCAT': r'percentage of saturation',
    'SMAP': r'm³/m³',
    'ERA5': r'm³/m³',
    'ERA5_LAND': r'm³/m³',
    'ESA_CCI_SM_active': r'percentage of saturation',
    'ESA_CCI_SM_combined': r'm³/m³',
    'ESA_CCI_SM_passive': r'm³/m³',
    'SMOS': r'm³/m³',
}

# label name for all metrics
_metric_name = {  # from /qa4sm/validator/validation/globals.py
    'R': 'Pearson\'s r',
    'p_R': 'Pearson\'s r p-value',
    'rho': 'Spearman\'s rho',
    'p_rho': 'Spearman\'s rho p-value',
    'RMSD': 'Root-mean-square deviation',
    'BIAS': 'Bias (difference of means)',
    'n_obs': '# observations',
    'urmsd': 'Unbiased root-mean-square deviation',
    'RSS': 'Residual sum of squares',
    'tau': 'Kendall rank correlation',
    'p_tau': 'Kendall tau p-value',
    'mse': 'Mean square error',
    'mse_corr': 'Mean square error correlation',
    'mse_bias': 'Mean square error bias',
    'mse_var': 'Mean square error variance',
    'snr': 'Signal-to-noise ratio',
    'err_std': 'Error standard deviation',
    'beta': 'TC scaling coefficient',
}

# === pretty names for datasets ===
# fallback for dataset pretty names in case they are not in the metadata
_dataset_pretty_names = {  # from qa4sm\validator\fixtures\datasets.json
    'ISMN': r'ISMN',
    'C3S': r'C3S',
    'GLDAS': r'GLDAS',
    'ASCAT': r'H-SAF ASCAT SSM CDR',
    'SMAP': r'SMAP level 3',
    'ERA5': r'ERA5',
    'ERA5_LAND': r'ERA5-Land',
    'ESA_CCI_SM_active': r'ESA CCI SM active',
    'ESA_CCI_SM_combined': r'ESA CCI SM combined',
    'ESA_CCI_SM_passive': r'ESA CCI SM passive',
    'SMOS': r'SMOS IC',
    'CGLS_CSAR_SSM1km': r'CGLS S1 SSM',
    'CGLS_SCATSAR_SWI1km': r'CGLS SCATSAR SWI',
}

# fallback for dataset __version pretty names in case they are not in the metadata
_dataset_version_pretty_names = {  # from qa4sm\validator\fixtures\versions.json
    "C3S_V201706": "v201706",
    "C3S_V201812": "v201812",
    "C3S_V201912": "v201912",
    "SMAP_V5_PM": "v5 PM/ascending",
    "SMAP_V5_AM": "v5 AM/descending",
    "ASCAT_H113": "H113",
    "ISMN_V20180712_TEST": "20180712 testset",
    "ISMN_V20180712_MINI": "20180712 mini testset",
    "ISMN_V20180830_GLOBAL": "20180830 global",
    "ISMN_V20190222_GLOBAL": "20190222 global",
    "ISMN_V20191211_GLOBAL": "20191211 global",
    "GLDAS_NOAH025_3H_2_1": "NOAH025 3H.2.1",
    "GLDAS_TEST": "TEST",
    "ESA_CCI_SM_C_V04_4": "v04.4",
    "ESA_CCI_SM_A_V04_4": "v04.4",
    "ESA_CCI_SM_P_V04_4": "v04.4",
    "ESA_CCI_SM_C_V04_5": "v04.5",
    "ESA_CCI_SM_A_V04_5": "v04.5",
    "ESA_CCI_SM_P_V04_5": "v04.5",
    "SMOS_105_ASC": "V.105 Ascending",
    "SMOS_105_DES": "V.105 Descending",
    "ERA5_test": " ERA5 test",
    "ERA5_20190613": "v20190613",
    "ERA5_LAND_V20190904" : "v20190904",
    "ERA5_LAND_TEST": "ERA5-Land test",
    "CGLS_CSAR_SSM1km_V1_1": "v1_1",
    "CGLS_SCATSAR_SWI1km_V1_0": "v1_0",
}

# label format for all metrics for HTML rendering
_metric_description_HTML = {  # from /qa4sm/validator/validation/graphics.py
    'R': ' [-]',
    'p_R': ' [-]',
    'rho': ' [-]',
    'p_rho': ' [-]',
    'tau': ' [-]',
    'p_tau': ' [-]',
    'RMSD': ' [{}]',
    'BIAS': ' [{}]',
    'n_obs': ' ',
    'urmsd': ' [{}]',
    'RSS': ' [({})²]',
    'mse': ' [({})²]',
    'mse_corr': ' [({})²]',
    'mse_bias': ' [({})²]',
    'mse_var': ' [({})²]',
    'snr': ' [db]',
    'err_std': ' [{}]',
    'beta': ' [{}]',
}

# units for all datasets for HTML rendering
_metric_units_HTML = {  # from /qa4sm/validator/validation/graphics.py
    'ISMN': 'm³/m³',
    'C3S': 'm³/m³',
    'GLDAS': 'm³/m³',
    'ASCAT': '% sat',
    'SMAP': 'm³/m³',
    'ERA5': 'm³/m³',
    'ERA5_LAND': 'm³/m³',
    'ESA_CCI_SM_active': '% sat',
    'ESA_CCI_SM_combined': 'm³/m³',
    'ESA_CCI_SM_passive': 'm³/m³',
    'SMOS': 'm³/m³',
    'CGLS_CSAR_SSM1km': '% sat',
    'CGLS_SCATSAR_SWI1km': '% sat',
}
