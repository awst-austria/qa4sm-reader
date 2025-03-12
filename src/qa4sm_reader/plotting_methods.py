# -*- coding: utf-8 -*-
"""
Contains helper functions for plotting qa4sm results.
"""
from logging import handlers
from qa4sm_reader import globals
from qa4sm_reader.exceptions import PlotterError
from qa4sm_reader.handlers import ClusteredBoxPlotContainer, CWContainer
from qa4sm_reader.utils import note

import numpy as np
import pandas as pd
import os.path

from typing import Union, List, Tuple, Dict, Optional, Any
import copy

import seaborn as sns
import matplotlib
import matplotlib.axes
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, PathPatch

from cartopy import config as cconfig
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs

from pygeogrids.grids import BasicGrid, genreg_grid
from shapely.geometry import Polygon, Point

import warnings
import os
from collections import namedtuple

cconfig['data_dir'] = os.path.join(os.path.dirname(__file__), 'cartopy')


def _float_gcd(a, b, atol=1e-08):
    "Greatest common divisor (=groesster gemeinsamer teiler)"
    while abs(b) > atol:
        a, b = b, a % b
    return a


def _get_grid(a):
    "Find the stepsize of the grid behind a and return the parameters for that grid axis."
    a = np.unique(a)  # get unique values and sort
    das = np.unique(np.diff(a))  # get unique stepsizes and sort
    da = das[0]  # get smallest stepsize
    for d in das[1:]:  # make sure, all stepsizes are multiple of da
        da = _float_gcd(d, da)
    a_min = a[0]
    a_max = a[-1]
    len_a = int((a_max - a_min) / da + 1)
    return a_min, a_max, da, len_a


def _get_grid_for_irregulars(a, grid_stepsize):
    "Find the stepsize of the grid behind a for datasets with predeifned grid stepsize, and return the parameters for that grid axis."
    a = np.unique(a)
    a_min = a[0]
    a_max = a[-1]
    da = grid_stepsize
    len_a = int((a_max - a_min) / da + 1)
    return a_min, a_max, da, len_a


def _value2index(a, a_min, da):
    "Return the indexes corresponding to a. a and the returned index is a numpy array."
    return ((a - a_min) / da).astype('int')


def _format_floats(x):
    """Format floats in the statistsics table"""
    if isinstance(x, float):
        if abs(x) < 0.000001:
            return "0"
        elif 0.1 < abs(x) < 1e3:
            return np.format_float_positional(x, precision=2)
        else:
            return np.format_float_scientific(x, precision=2)
    else:
        return x


def oversample(lon, lat, data, extent, dx, dy):
    """Sample to regular grid"""
    other = BasicGrid(lon, lat)
    reg_grid = genreg_grid(dx,
                           dy,
                           minlat=extent[2],
                           maxlat=extent[3],
                           minlon=extent[0],
                           maxlon=extent[1])
    max_dist = dx * 111 * 1000  # a mean distance for one degree it's around 111 km
    lut = reg_grid.calc_lut(other, max_dist=max_dist)
    img = np.ma.masked_where(lut == -1, data[lut])
    img[np.isnan(img)] = np.ma.masked

    return img.reshape(-1, reg_grid.shape[1]), reg_grid


def geotraj_to_geo2d(df, index=globals.index_names, grid_stepsize=None):
    """
    Converts geotraj (list of lat, lon, value) to a regular grid over lon, lat.
    The values in df needs to be sampled from a regular grid, the order does not matter.
    When used with plt.imshow(), specify data_extent to make sure,
    the pixels are exactly where they are expected.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'lat', 'lon' and 'var' Series.
    index : tuple, optional
        Tuple containing the names of lattitude and longitude index. Usually ('lat','lon')
        The default is globals.index_names
    grid_stepsize : None or float, optional
        angular grid stepsize to prepare a regular grid for plotting

    Returns
    -------
    zz : numpy.ndarray
        array holding the gridded values. When using plt.imshow, specify origin='lower'.
        [0,0] : llc (lower left corner)
        first coordinate is longitude.
    data_extent : tuple
        (x_min, x_max, y_min, y_max) in Data coordinates.
    origin : string
        'upper' or 'lower' - define how the plot should be oriented, for irregular grids it should return 'upper'
    """
    xx = df.index.get_level_values(index[1])  # lon
    yy = df.index.get_level_values(index[0])  # lat

    if grid_stepsize not in ['nan', None]:
        x_min, x_max, dx, len_x = _get_grid_for_irregulars(xx, grid_stepsize)
        y_min, y_max, dy, len_y = _get_grid_for_irregulars(yy, grid_stepsize)
        data_extent = (x_min - dx / 2, x_max + dx / 2, y_min - dy / 2,
                       y_max + dy / 2)
        zz, grid = oversample(xx, yy, df.values, data_extent, dx, dy)
        origin = 'upper'
    else:
        x_min, x_max, dx, len_x = _get_grid(xx)
        y_min, y_max, dy, len_y = _get_grid(yy)
        ii = _value2index(yy, y_min, dy)
        jj = _value2index(xx, x_min, dx)
        zz = np.full((len_y, len_x), np.nan, dtype=np.float64)
        zz[ii, jj] = df
        data_extent = (x_min - dx / 2, x_max + dx / 2, y_min - dy / 2,
                       y_max + dy / 2)
        origin = 'lower'

    return zz, data_extent, origin


def get_value_range(ds,
                    metric=None,
                    force_quantile=False,
                    quantiles=[0.025, 0.975],
                    diff_map=False):
    """
    Get the value range (v_min, v_max) from globals._metric_value_ranges
    If the range is (None, None), a symmetric range around 0 is created,
    showing at least the symmetric <quantile> quantile of the values.
    if force_quantile is True, the quantile range is used.

    Parameters
    ----------
    ds : pd.DataFrame or pd.Series
        Series holding the values
    metric : str , optional (default: None)
        name of the metric (e.g. 'R'). None equals to force_quantile=True.
    force_quantile : bool, optional
        always use quantile, regardless of globals.
        The default is False.
    quantiles : list, optional
        quantile of data to include in the range.
        The default is [0.025,0.975]
    diff_map : bool, default is False
        Whether the colorbar is for a difference plot

    Returns
    -------
    v_min : float
        lower value range of plot.
    v_max : float
        upper value range of plot.
    """
    if metric == None:
        force_quantile = True

    ranges = globals._metric_value_ranges
    if not force_quantile:  # try to get range from globals
        try:
            v_min = ranges[metric][0]
            v_max = ranges[metric][1]
            if (v_min is None and v_max is None
                ):  # get quantile range and make symmetric around 0.
                v_min, v_max = get_quantiles(ds, quantiles)
                v_max = max(
                    abs(v_min),
                    abs(v_max))  # make sure the range is symmetric around 0
                v_min = -v_max
            elif v_min is None:
                v_min = get_quantiles(ds, quantiles)[0]
            elif v_max is None:
                v_max = get_quantiles(ds, quantiles)[1]
            else:  # v_min and v_max are both determinded in globals
                pass
        except KeyError:  # metric not known, fall back to quantile
            force_quantile = True
            warnings.warn('The metric \'{}\' is not known. \n'.format(metric) + \
                          'Could not get value range from globals._metric_value_ranges\n' + \
                          'Computing quantile range \'{}\' instead.\n'.format(str(quantiles)) +
                          'Known metrics are: \'' + \
                          '\', \''.join([metric for metric in ranges]) + '\'')

    if force_quantile:  # get quantile range
        v_min, v_max = get_quantiles(ds, quantiles)
        # adjust range based on the difference values in the map
        if diff_map:
            extreme = max([abs(v) for v in get_quantiles(ds, quantiles)])
            v_min, v_max = -extreme, extreme

    return v_min, v_max


def get_quantiles(ds, quantiles) -> tuple:
    """
    Gets lower and upper quantiles from pandas.Series or pandas.DataFrame

    Parameters
    ----------
    ds : (pandas.Series | pandas.DataFrame)
        Input values.
    quantiles : list
        quantile of values to include in the range

    Returns
    -------
    v_min : float
        lower quantile.
    v_max : float
        upper quantile.

    """
    q = ds.quantile(quantiles)
    if isinstance(ds, pd.Series):
        return q.iloc[0], q.iloc[1]
    elif isinstance(ds, pd.DataFrame):
        return min(q.iloc[0]), max(q.iloc[1])
    else:
        raise TypeError(
            "Inappropriate argument type. 'ds' must be pandas.Series or pandas.DataFrame."
        )


def get_plot_extent(df, grid_stepsize=None, grid=False) -> tuple:
    """
    Gets the plot_extent from the values. Uses range of values and
    adds a padding fraction as specified in globals.map_pad

    Parameters
    ----------
    grid : bool
        whether the values in df is on a equally spaced grid (for use in mapplot)
    df : pandas.DataFrame
        Plot values.

    Returns
    -------
    extent : tuple | list
        (x_min, x_max, y_min, y_max) in Data coordinates.

    """
    lat, lon, gpi = globals.index_names
    if grid and grid_stepsize in ['nan', None]:
        # todo: problem if only single lon/lat point is present?
        x_min, x_max, dx, len_x = _get_grid(df.index.get_level_values(lon))
        y_min, y_max, dy, len_y = _get_grid(df.index.get_level_values(lat))
        extent = [
            x_min - dx / 2., x_max + dx / 2., y_min - dx / 2., y_max + dx / 2.
        ]
    elif grid and grid_stepsize:
        x_min, x_max, dx, len_x = _get_grid_for_irregulars(
            df.index.get_level_values(lon), grid_stepsize)
        y_min, y_max, dy, len_y = _get_grid_for_irregulars(
            df.index.get_level_values(lat), grid_stepsize)
        extent = [
            x_min - dx / 2., x_max + dx / 2., y_min - dx / 2., y_max + dx / 2.
        ]
    else:
        extent = [
            df.index.get_level_values(lon).min(),
            df.index.get_level_values(lon).max(),
            df.index.get_level_values(lat).min(),
            df.index.get_level_values(lat).max()
        ]
    dx = extent[1] - extent[0]
    dy = extent[3] - extent[2]
    # set map-padding around values to be globals.map_pad percent of the smaller dimension
    padding = min(dx, dy) * globals.map_pad / (1 + globals.map_pad)
    extent[0] -= padding
    extent[1] += padding
    extent[2] -= padding
    extent[3] += padding
    if extent[0] < -180:
        extent[0] = -180
    if extent[1] > 180:
        extent[1] = 180
    if extent[2] < -90:
        extent[2] = -90
    if extent[3] > 90:
        extent[3] = 90

    return extent


def init_plot(figsize,
              dpi,
              add_cbar=None,
              projection=None,
              fig_template=None) -> tuple:
    """Initialize mapplot"""
    if not projection:
        projection = globals.crs

    if fig_template is None:
        # fig, ax_main = plt.subplots(figsize=figsize, dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig = fig_template.fig
        ax_main = fig_template.ax_main

    if add_cbar:
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[19, 1])
        ax_main = fig.add_subplot(gs[0], projection=projection)
        cax = fig.add_subplot(gs[1])
    else:
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax_main = fig.add_subplot(gs[0], projection=projection)
        cax = None

    return fig, ax_main, cax


def get_extend_cbar(metric):
    """
    Find out whether the colorbar should extend, based on globals._metric_value_ranges[metric]

    Parameters
    ----------
    metric : str
        metric used in plot

    Returns
    -------
    str
        one of ['neither', 'min', 'max', 'both'].
    """
    vrange = globals._metric_value_ranges[metric]
    if vrange[0] is None:
        if vrange[1] is None:
            return 'both'
        else:
            return 'min'
    else:
        if vrange[1] is None:
            return 'max'
        else:
            return 'neither'


def style_map(
    ax,
    plot_extent,
    add_grid=True,
    map_resolution=globals.naturalearth_resolution,
    add_topo=False,
    add_coastline=True,
    add_land=True,
    add_borders=True,
    add_us_states=False,
    grid_intervals=globals.grid_intervals,
):
    """Parameters to style the mapplot"""
    ax.set_extent(plot_extent, crs=globals.data_crs)
    ax.spines["geo"].set_linewidth(0.4)
    if add_grid:
        # add gridlines. Bcs a bug in cartopy, draw girdlines first and then grid labels.
        # https://github.com/SciTools/cartopy/issues/1342
        try:
            grid_interval = max(
                (plot_extent[1] - plot_extent[0]),
                (plot_extent[3] - plot_extent[2]
                 )) / 5  # create apprx. 5 gridlines in the bigger dimension
            if grid_interval <= min(grid_intervals):
                raise RuntimeError
            grid_interval = min(
                grid_intervals, key=lambda x: abs(x - grid_interval)
            )  # select the grid spacing from the list which fits best
            gl = ax.gridlines(crs=globals.data_crs,
                              draw_labels=False,
                              linewidth=0.5,
                              color='grey',
                              linestyle='--',
                              zorder=3)  # draw only gridlines.
            # todo: this can slow the plotting down!!
            xticks = np.arange(-180, 180.001, grid_interval)
            yticks = np.arange(-90, 90.001, grid_interval)
            gl.xlocator = mticker.FixedLocator(xticks)
            gl.ylocator = mticker.FixedLocator(yticks)
        except RuntimeError:
            pass
        else:
            try:  # drawing labels fails for most projections
                gltext = ax.gridlines(crs=globals.data_crs,
                                      draw_labels=True,
                                      linewidth=0.5,
                                      color='grey',
                                      alpha=0.,
                                      linestyle='-',
                                      zorder=4)  # draw only grid labels.
                xticks = xticks[(xticks >= plot_extent[0])
                                & (xticks <= plot_extent[1])]
                yticks = yticks[(yticks >= plot_extent[2])
                                & (yticks <= plot_extent[3])]
                gltext.xformatter = LONGITUDE_FORMATTER
                gltext.yformatter = LATITUDE_FORMATTER
                gltext.top_labels = False
                gltext.right_labels = False
                gltext.xlocator = mticker.FixedLocator(xticks)
                gltext.ylocator = mticker.FixedLocator(yticks)
            except RuntimeError as e:
                print("No tick labels plotted.\n" + str(e))
    if add_topo:
        ax.stock_img()
    if add_coastline:
        coastline = cfeature.NaturalEarthFeature('physical',
                                                 'coastline',
                                                 map_resolution,
                                                 edgecolor='black',
                                                 facecolor='none')
        ax.add_feature(coastline, linewidth=0.4, zorder=3)
    if add_land:
        land = cfeature.NaturalEarthFeature('physical',
                                            'land',
                                            map_resolution,
                                            edgecolor='none',
                                            facecolor='white')
        ax.add_feature(land, zorder=1)
    if add_borders:
        borders = cfeature.NaturalEarthFeature('cultural',
                                               'admin_0_countries',
                                               map_resolution,
                                               edgecolor='black',
                                               facecolor='none')
        ax.add_feature(borders, linewidth=0.5, zorder=3)
    if add_us_states:
        ax.add_feature(cfeature.STATES, linewidth=0.1, zorder=3)

    return ax


@note(
    "DeprecationWarning: The function `qa4sm_reader.plotting_methods.make_watermark()` is deprecated and will be removed in the next release. Use `qa4sm_reader.plotting_methods.add_logo_to_figure` instead to add a logo."
)
def make_watermark(fig,
                   placement=globals.watermark_pos,
                   for_map=False,
                   offset=0.03,
                   for_barplot=False,
                   fontsize=globals.watermark_fontsize):
    """
    Adds a watermark to fig and adjusts the current axis to make sure there
    is enough padding around the watermarks.
    Padding can be adjusted in globals.watermark_pad.
    Fontsize can be adjusted in globals.watermark_fontsize.
    plt.tight_layout needs to be called prior to make_watermark,
    because tight_layout does not take into account annotations.
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    placement : str
        'top' : places watermark in top right corner
        'bottom' : places watermark in bottom left corner
    for_map : bool
        True if watermark is for mapplot
    for_barplot : bool
        True if watermark is for barplot
    """
    # ax = fig.gca()
    # pos1 = ax.get_position() #fraction of figure
    pad = globals.watermark_pad
    height = fig.get_size_inches()[1]
    offset = offset + ((
        (fontsize + pad) / globals.matplotlib_ppi) / height) * 2.2
    if placement == 'top':
        plt.annotate(
            globals.watermark,
            xy=[0.5, 1],
            xytext=[-pad, -pad],
            fontsize=fontsize,
            color='white',  #TODO! change back to grey
            horizontalalignment='center',
            verticalalignment='top',
            xycoords='figure fraction',
            textcoords='offset points')
        top = fig.subplotpars.top
        fig.subplots_adjust(top=top - offset)

    elif for_map or for_barplot:
        if for_barplot:
            plt.suptitle(
                globals.watermark,
                color='white',  #TODO! change back to grey
                fontsize=fontsize,
                x=-0.07,
                y=0.5,
                va='center',
                rotation=90)
        else:
            plt.suptitle(
                globals.watermark,
                color='white',  #TODO! change back to grey
                fontsize=fontsize,
                y=0,
                ha='center')

    elif placement == 'bottom':
        plt.annotate(
            globals.watermark,
            xy=[0.5, 0],
            xytext=[pad, pad],
            fontsize=fontsize,
            color='white',  #TODO! change back to grey
            horizontalalignment='center',
            verticalalignment='bottom',
            xycoords='figure fraction',
            textcoords='offset points')
        bottom = fig.subplotpars.bottom
        if not for_map:
            fig.subplots_adjust(bottom=bottom + offset)
    else:
        raise NotImplementedError


#$$
Offset = namedtuple('offset',
                    ['x', 'y'])  # helper for offset in add_logo_to_figure


def add_logo_to_figure(
        fig: matplotlib.figure.Figure,
        logo_path: Optional[str] = globals.watermark_logo_pth,
        position: Optional[str] = globals.watermark_logo_position,
        offset: Optional[Union[Tuple, Offset]] = (0., -0.15),
        scale: Optional[float] = 0.15) -> None:
    """
    Add a logo to an existing figure. This is done by creating an additional axis in the figure, at the location\
        specified by `position`. The logo is then placed on this axis.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        The figure to add the logo to. The figure should have at least one axis, otherwise an axis is created.z

    logo_path: Optional[str]
        Path to the logo image. If the path does not exist, a warning is raised and the function returns. Default is\
            `globals.watermark_logo_pth`.

    position: Optional[str]
        The position of the logo in the figure. Valid values are 'lower_left', 'lower_center', 'lower_right',\
            'upper_left', 'upper_center', 'upper_right'. Default is `globals.watermark_logo_position`.

    offset: Optional[Tuple | Offset]
        Offset of the logo from the right edge of the subplot (right lower corner of the main plot).\
            The first value is the x-offset, the second value is the y-offset. Default is (0., 0).

    scale: Optional[float]
        Scale of the logo relative to the figure height (= fraction of figure height). Valid values are (0, 1].\
            Default is 0.15.

    Returns
    -------
    None
    """

    if not fig.get_axes():
        warnings.warn("No axes found in the figure. Creating a new one.")
        fig.add_subplot(111)

    if not os.path.exists(logo_path):
        warnings.warn(
            f"No logo found at the specified path: '{logo_path}'. Skipping logo addition."
        )
        print(
            f"No logo found at the specified path: '{logo_path}'. Skipping logo addition."
        )
        return

    with cbook.get_sample_data(logo_path) as file:
        im = mpimg.imread(file)

    # Get the dimensions of the image
    height, width, _ = im.shape

    fig_height_pixels = fig.get_figheight() * fig.dpi

    logo_height_pixels = scale * fig_height_pixels
    logo_width_pixels = width * logo_height_pixels / height

    # Convert back to figure coordinates
    logo_width_fig = logo_width_pixels / fig.dpi

    if not isinstance(offset, Offset):
        offset = Offset(*offset)

    if 'left' in position:
        left = 1 - (logo_width_fig) + offset.x
    elif 'center' in position:
        left = 0.5 - (logo_width_fig / 2) + offset.x
    elif 'right' in position:  # 'right' in position
        left = 0 + offset.x

    if 'lower' in position:
        bottom = offset.y
    elif 'upper' in position:  # 'upper' in position
        bottom = 1 - offset.y

    # Define the new position of ax_logo
    # [left, bottom, width, height]
    ax_logo_pos = [left, bottom, logo_width_fig, scale]

    # Add a new axis to the figure at the position of ax_logo to house the logo
    ax_logo = fig.add_axes(ax_logo_pos)
    ax_logo.imshow(im)

    # Hide the axis
    ax_logo.axis('off')


def _make_cbar(fig,
               im,
               cax,
               ref_short: str,
               metric: str,
               label=None,
               diff_map=False,
               scl_short=None):
    """
    Make colorbar to use in plots

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure of plot
    im: AxesImage
        from method Axes.imshow()
    cax: axes.SubplotBase
        from fig.add_subplot
    ref_short: str
        name of ref dataset
    scl_short : str, default is None
        name of scaling dataset
    metric: str
        name of metric
    label: str
        label to describe the colorbar
    diff_map : bool, default is False
        Whether the colorbar is for a difference plot

    """
    if label is None:
        label = globals._metric_name[metric] + \
                globals._metric_description[metric].format(
                    globals.get_metric_units(ref_short)
                )
        if scl_short:
            label = globals._metric_name[metric] + \
                    globals._metric_description[metric].format(
                        globals.get_metric_units(scl_short)
                    )

    extend = get_extend_cbar(metric)
    if diff_map:
        extend = "both"
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
    cbar.set_label(label, weight='normal')
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(width=0.4)

    return fig, im, cax


def _CI_difference(fig, ax, ci):
    """
    Insert the median value of the upper and lower CI difference

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure with CIs
    ci: list
        list of upper and lower ci dataframes
    """
    lower_pos = []
    for ax in fig.axes:
        n = 0
        # iterating through axes artists:
        for c in ax.get_children():
            # searching for PathPatches
            if isinstance(c, PathPatch):
                # different width whether it's the metric or the CIs
                if n in np.arange(0, 100, 3):
                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    lower_pos.append(xmin)
                n += 1
    for ci_df, xmin in zip(ci, lower_pos):
        diff = ci_df["upper"] - ci_df["lower"]
        ci_range = float(diff.mean())
        ypos = float(ci_df["lower"].min())
        ax.annotate("Mean CI\nRange:\n {:.2g}".format(ci_range),
                    xy=(xmin - 0.2, ypos),
                    horizontalalignment="center")


def _add_dummies(df: pd.DataFrame, to_add: int) -> list:
    """
    Add empty columns in dataframe to avoid error in matplotlib when not all boxplot groups have the same
    number of values
    """
    for n, col in enumerate(np.arange(to_add)):
        # add columns while avoiding name clashes
        df[str(n)] = np.nan

    return df


def patch_styling(box_dict, facecolor) -> None:
    """Define style of the boxplots"""
    for n, (patch,
            median) in enumerate(zip(box_dict["boxes"], box_dict["medians"])):
        patch.set(color="grey", facecolor=facecolor, linewidth=1.6, alpha=0.7)
        median.set(color="grey", linewidth=1.6)
    for (whis, caps) in zip(box_dict["whiskers"], box_dict["caps"]):
        whis.set(color="grey", linewidth=1.6)
        caps.set(color="grey", linewidth=1.6)


def _box_stats(ds: pd.Series,
               med: bool = True,
               iqrange: bool = True,
               count: bool = True) -> str:
    """
    Create the metric part with stats of the box (axis) caption

    Parameters
    ----------
    ds: pd.Series
        data on which stats are found
    med: bool
    iqrange: bool
    count: bool
        statistics

    Returns
    -------
    stats: str
        caption with summary stats
    """
    # interquartile range
    iqr = ds.quantile(q=[0.75, 0.25]).diff()
    iqr = abs(float(iqr.loc[0.25]))

    met_str = []
    if med:
        met_str.append('Median: {:.3g}'.format(ds.median()))
    if iqrange:
        met_str.append('IQR: {:.3g}'.format(iqr))
    if count:
        met_str.append('N: {:d}'.format(ds.count()))
    stats = '\n'.join(met_str)

    return stats


def boxplot(
    df,
    ci=None,
    label=None,
    figsize=None,
    dpi=100,
    spacing=0.35,
    axis=None,
    **plotting_kwargs,
) -> tuple:
    """
    Create a boxplot_basic from the variables in df.
    The box shows the quartiles of the dataset while the whiskers extend
    to show the rest of the distribution, except for points that are
    determined to be “outliers” using a method that is a function of
    the inter-quartile range.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'lat', 'lon' and (multiple) 'var' Series.
    ci : list
        list of Dataframes containing "upper" and "lower" CIs
    label : str, optional
        Label of the y axis, describing the metric. The default is None.
    figsize : tuple, optional
        Figure size in inches. The default is globals.map_figsize.
    dpi : int, optional
        Resolution for raster graphic output. The default is globals.dpi.
    spacing : float, optional.
        Space between the central boxplot and the CIs. Default is 0.3
    axis : matplotlib Axis obj.
        if provided, the plot will be shown on it

    Returns
    -------
    fig : matplotlib.figure.Figure
        the boxplot
    ax : matplotlib.axes.Axes
    """
    values = df.copy()
    center_pos = np.arange(len(values.columns)) * 2
    # make plot

    ax = axis
    if axis is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = None
    ticklabels = values.columns
    # styling of the boxes
    kwargs = {"patch_artist": True, "return_type": "dict"}
    for key, value in plotting_kwargs.items():
        kwargs[key] = value
    # changes necessary to have confidence intervals in the plot
    # could be an empty list or could be 'None', if de-selected from the kwargs
    if ci:
        upper, lower = [], []
        for n, intervals in enumerate(ci):
            lower.append(intervals["lower"])
            upper.append(intervals["upper"])
        lower = _add_dummies(
            pd.concat(lower, ignore_index=True, axis=1),
            len(center_pos) - len(ci),
        )
        upper = _add_dummies(
            pd.concat(upper, ignore_index=True, axis=1),
            len(center_pos) - len(ci),
        )
        low = lower.boxplot(positions=center_pos - spacing,
                            showfliers=False,
                            widths=0.15,
                            ax=ax,
                            **kwargs)
        up = upper.boxplot(positions=center_pos + spacing,
                           showfliers=False,
                           widths=0.15,
                           ax=ax,
                           **kwargs)
        patch_styling(low, 'skyblue')
        patch_styling(up, 'tomato')

    if not 'positions' in kwargs:
        positions = center_pos
    else:
        positions = kwargs['positions']
        del kwargs['positions']

    if not 'widths' in kwargs:
        widths = 0.3
    else:
        widths = kwargs['widths']
        del kwargs['widths']

    cen = values.boxplot(positions=positions,
                         showfliers=False,
                         widths=widths,
                         ax=ax,
                         **kwargs)
    patch_styling(cen, 'white')

    if ci:
        low_ci = Patch(color='skyblue', alpha=0.7, label='Lower CI')
        up_ci = Patch(color='tomato', alpha=0.7, label='Upper CI')
        # _CI_difference(fig, ax, ci)
        ax.legend(handles=[low_ci, up_ci], fontsize=8, loc="best")
    # provide y label
    if label is not None:
        plt.ylabel(label, weight='normal')
    ax.set_xticks(positions)
    ax.set_xticklabels(ticklabels)
    ax.tick_params(labelsize=globals.tick_size)
    ax.grid(axis='x')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return fig, ax


def _replace_status_values(ser):
    """
    Replace values in series to plot less categories in the error plots,
    according to globals.status_replace dict.

    Parameters
    ----------
    ser : pandas.Series
        Series containing 'lat', 'lon' and status values.

    Returns
    -------
    ser : pandas.Series
    """
    assert type(ser) == pd.Series
    for val in set(ser.values):
        # all new error codes replaced with -1
        if val not in globals.status.keys():
            ser = ser.replace(to_replace=val, value=-1)
        if val in globals.status_replace.keys():
            ser = ser.replace(to_replace=val,
                              value=globals.status_replace[val])
    return ser


def barplot(
    df,
    label=None,
    figsize=None,
    dpi=100,
    axis=None,
) -> tuple:
    """
    Create a barplot from the validation errors in df.
    The bars show the numbers of errors that occured during
    the validation between two or three (in case of triple
    collocation) datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'lat', 'lon' and (multiple) 'var' Series.
    label : str, optional
        Label of the y axis, describing the metric. The default is None.
    figsize : tuple, optional
        Figure size in inches. The default is globals.map_figsize.
    dpi : int, optional
        Resolution for raster graphic output. The default is globals.dpi.
    axis : matplotlib Axis obj.
        if provided, the plot will be shown on it

    Returns
    -------
    fig : matplotlib.figure.Figure
        the boxplot
    ax : matplotlib.axes.Axes
    """

    ax = axis
    if axis is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = None

    values = df.copy()
    values = values[[values.keys()[0]]]
    values.dropna(inplace=True)
    status_dict = globals.status
    values[values.keys()[0]] = _replace_status_values(values[values.keys()[0]])
    vals = sorted(list(set(values[values.keys()[0]])))

    tick_entries = [status_dict[x] for x in vals]
    tick_labels = [
        "-\n".join([entry[i:i + 18] for i in range(0, len(entry), 18)])
        for entry in tick_entries
    ]
    color = [globals.get_status_colors().colors[int(x) + 1] for x in vals]
    values[values.keys()[0]].value_counts().sort_index().plot.bar(ax=ax,
                                                                  color=color)

    ax.tick_params(labelsize=globals.tick_size)
    ax.grid(axis='y')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels(tick_labels, rotation=45)

    plt.ylabel(label, weight='normal')

    return fig, ax


# TODO: test?
def resize_bins(sorted, nbins):
    """Resize the bins for "continuous" metadata types"""
    bin_edges = np.linspace(0, 100, nbins + 1)
    p_rank = 100.0 * (np.arange(sorted.size) + 0.5) / sorted.size
    # use +- 1 to make sure nothing falls outside bins
    bin_edges = np.interp(bin_edges,
                          p_rank,
                          sorted,
                          left=sorted[0] - 1,
                          right=sorted[-1] + 1)
    bin_values = np.digitize(sorted, bin_edges)
    unique_values, counts = np.unique(bin_values, return_counts=True)
    bin_size = max(counts)

    return bin_values, unique_values, bin_size


def bin_continuous(
    df: pd.DataFrame,
    metadata_values: pd.DataFrame,
    meta_key: str,
    nbins=4,
    min_size=5,
    **kwargs,
) -> Union[dict, None]:
    """
    Subset the continuous metadata types

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of the values to plot
    metadata_values : pd.DataFrame
        metadata values
    meta_key : str
        name of the metadata
    nbins : int. Default is 4.
        Bins to divide the metadata range into
    min_size : int. Default is 5
        Minimum number of values to have in a bin
    kwargs: dict
        Keyword arguments for specific metadata types

    Returns
    -------
    binned: dict
        dictionary with metadata subsets as keys
    """
    meta_units = globals.metadata[meta_key][3]
    meta_range = metadata_values[meta_key].to_numpy()
    sorted = np.sort(meta_range)
    if len(meta_range) < min_size:
        raise ValueError(
            "There are too few points per metadata to generate the boxplots. "
            f"You can set 'min_size' (now at {min_size})"
            "to a lower value to allow for smaller samples.")
    bin_values, unique_values, bin_size = resize_bins(sorted, nbins)
    # adjust bins to have the specified number of bins if possible, otherwise enough valoues per bin
    while bin_size < min_size:
        nbins -= 1
        bin_values, unique_values, bin_size = resize_bins(sorted, nbins)

    # use metadata to sort dataframe
    df = pd.concat([df, metadata_values], axis=1).sort_values(meta_key)
    df.drop(columns=meta_key, inplace=True)
    # put binned data in dataframe
    binned = {}
    for bin in unique_values:
        bin_index = np.where(bin_values == bin)
        bin_sorted = sorted[bin_index]
        bin_df = df.iloc[bin_index]
        bin_label = "{:.2f}-{:.2f} {}".format(min(bin_sorted), max(bin_sorted),
                                              meta_units)
        if not all(col >= min_size for col in bin_df.count()):
            continue
        binned[bin_label] = bin_df
    # If too few points are available to make the plots
    if not binned:
        return None

    return binned


def bin_classes(
    df: pd.DataFrame,
    metadata_values: pd.DataFrame,
    meta_key: str,
    min_size=5,
    **kwargs,
) -> Union[dict, None]:
    """
    Subset the continuous metadata types

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of the values to plot
    metadata_values : pd.DataFrame
        metadata values
    meta_key : str
        name of the metadata
    min_size : int. Default is 5
        Minimum number of values to have in a bin
    kwargs: dict
        Keyword arguments for specific metadata types

    Returns
    -------
    binned: dict
        dictionary with metadata subsets as keys
    """
    classes_lut = globals.metadata[meta_key][1]
    grouped = metadata_values.applymap(lambda x: classes_lut[x])
    binned = {}
    for meta_class, meta_df in grouped.groupby(meta_key).__iter__():
        bin_df = df.loc[meta_df.index]
        if not all(col >= min_size for col in bin_df.count()):
            continue
        binned[meta_class] = bin_df

    # If too few points are available to make the plots
    if not binned:
        return None

    return binned


def bin_discrete(
    df: pd.DataFrame,
    metadata_values: pd.DataFrame,
    meta_key: str,
    min_size=5,
    **kwargs,
) -> Union[pd.DataFrame, None]:
    """
    Provide a formatted dataframe for discrete type metadata (e.g. station or network)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of the values to plot
    metadata_values : pd.DataFrame
        metadata values
    meta_key : str
        name of the metadata
    min_size : int. Default is 5
        Minimum number of values to have in a bin
    kwargs: dict
        Keyword arguments for specific metadata types

    Returns
    -------
    formatted: pd.DataFrame
        Dataframe formatted for seaborn plotting
    """
    groups = []
    for col in df.columns:
        group = pd.concat([df[col], metadata_values], axis=1)
        group.columns = ["values", meta_key]
        group["Dataset"] = col
        groups.append(group)
    grouped = pd.concat(groups, axis=0)
    formatted = []
    counts = grouped.groupby([meta_key, 'Dataset']).count()
    for meta, meta_df in grouped.groupby(meta_key).__iter__():
        filtered_df = meta_df.copy()
        # Filter rows based on whether their (network, dataset) combination meets the threshold
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: counts.loc[(meta, row['Dataset'])]['values'] >= min_size,
            axis=1
        )]
        if len(filtered_df):
            formatted.append(filtered_df)
    # If too few points are available to make the plots
    if not formatted:
        return None
    else:
        formatted = pd.concat(formatted)
        # return None as no CI data is needed for this plot
        return formatted


def bin_function_lut(type):
    """Lookup table between the metadata type and the binning function"""
    lut = {
        "continuous": bin_continuous,
        "discrete": bin_discrete,
        "classes": bin_classes,
    }
    if type not in lut.keys():
        raise KeyError(
            "The type '{}' does not correspond to any binning function".format(
                type))

    return lut[type]


def _stats_discrete(df: pd.DataFrame, meta_key: str, stats_key: str) -> list:
    """Return list of stats by group, where groups are created with a specific key"""
    stats_list = []
    for _key, group in df.groupby(meta_key).__iter__():
        stats = _box_stats(group[stats_key])
        median = group[stats_key].median()
        stats_list.append((stats, median))

    return stats_list


def combine_soils(
    soil_fractions: dict,
    clay_fine: int = 35,
    clay_coarse: int = 20,
    sand_coarse: int = 65,
) -> pd.DataFrame:
    """
    Create a metadata granulometry classification based on 'coarse', 'medium' or 'fine' soil types. Uses
    the soil texture triangle diagram to transform the values.

    Parameters
    ----------
    soil_fractions: dict
        Dictionary with {'soil type (clay, sand or silt)': qa4sm_handlers.Metadata}
    clay_fine: int
        clay threshold above which the soil is fine
    clay_coarse: int
        clay threshold below which the soil can be coarse
    sand_coarse: int
        sand threshold above which the soil can be coarse

    Returns
    -------
    soil_combined: pd.DataFrame
        Dataframe with the new metadata type
    """
    # get thresholds on cartesian plane
    cf_y = clay_fine * np.sin(2 / 3 * np.pi)
    cc_y = clay_coarse * np.sin(2 / 3 * np.pi)
    sc_x = 100 - sand_coarse
    # transform values to cartesian
    x = soil_fractions["sand_fraction"].values.apply(lambda x: 100 - x)
    y = soil_fractions["clay_fraction"].values.apply(
        lambda x: x * np.sin(2 / 3 * np.pi))
    soil_combined = pd.concat([x, y], axis=1)
    soil_combined.columns = ["x", "y"]

    # function to calssify
    def sort_soil_type(row):
        if row["x"] < sc_x and row["y"] < cc_y:
            return "Coarse\ngran."
        elif cc_y < row["y"] < cf_y:
            return "Medium\ngran."
        else:
            return "Fine\ngran."

    soil_combined = soil_combined.apply(lambda row: sort_soil_type(row),
                                        axis=1).to_frame("soil_type")

    return soil_combined


def combine_depths(depth_dict: dict) -> pd.DataFrame:
    """
    Create a metadata entry for the instrument depth by finding the middle point between the upper and lower
    specified instrument depths

    Parameters
    ----------
    depth_dict: dict
        Dictionary with {'instrument_depthfrom/instrument_depthto': qa4sm_handlers.Metadata}

    Returns
    -------
    depths_combined: pd.DataFrame
        Dataframe with the new metadata type
    """
    depths_combined = []
    for key, obj in depth_dict.items():
        depths_combined.append(obj.values)

    depths_combined = pd.concat(depths_combined, axis=1)
    depths_combined = depths_combined.mean(axis=1).to_frame("instrument_depth")

    return depths_combined


def aggregate_subplots(to_plot: dict, funct, n_bars, common_y=None, **kwargs):
    """
    Aggregate multiple subplots into one image

    Parameters
    ----------
    to_plot: dict
        dictionary with the data to plot, of the shape 'title of the subplot': pd.Dataframe
        (or data format required by funct)
    funct: method
        function to create the individual subplots. Should have a parameter 'axis',
        where the plt.Axis can be given. Returns a tuple of (unit_height, unit_width)
    n_bars: int
        number of boxplot bars (one is central + confidence intervals)
    **kwargs: dict
        arguments to pass on to the plotting function

    Return
    ------
    fig, axes
    """
    sub_n = len(to_plot.keys())
    if sub_n == 1:
        for n, (bin_label, data) in enumerate(to_plot.items()):
            fig, axes = funct(df=data, **kwargs)
    elif sub_n > 1:
        # provide the figure and subplots
        rows = int(np.ceil(sub_n / 2))
        fig, axes = plt.subplots(rows, 2, sharey=True)
        for n, (bin_label, data) in enumerate(to_plot.items()):
            if n % 2 == 0:
                try:
                    ax = axes[int(n / 2), 0]
                except IndexError:  # If only two subplots, it is a 1-dimensional array
                    ax = axes[0]
            else:
                try:
                    ax = axes[int(n / 2), 1]
                except IndexError:
                    ax = axes[1]
            # Make sure funct has the correct parameters format
            if 'axis' not in funct.__code__.co_varnames:
                raise KeyError(
                    "'axis' should be in the parameters of the given function {}"
                    .format(funct))
            funct(df=data, axis=ax, **kwargs)
            ax.set_title(bin_label, fontdict={"fontsize": 10})
            if n != 0:
                ax.legend([], [], frameon=False)
        # eliminate extra subplot if odd number
        if rows * 2 > sub_n:
            fig.delaxes(axes[rows - 1, 1])

        plt.subplots_adjust(wspace=0.1, hspace=0.25)
        fig.set_figheight(globals.boxplot_height * (np.ceil(sub_n / 2) + 0.2))
        fig.set_figwidth(globals.boxplot_width * n_bars * 2)

        if common_y:
            fig.text(0.05, 0.5, common_y, va='center', rotation='vertical')

    return fig, axes


def bplot_multiple(to_plot, y_axis, n_bars, **kwargs) -> tuple:
    """
    Create subplots for each metadata category/range

    Parameters
    ----------
    to_plot : dict
        dictionary of {'bin name': Dataframe}
    y_axis : str
        Name of the x-axis
    n_bars : int or float
        Number of datasets/boxplot bars
    """
    # create plot with as many subplots as the dictionary keys
    n_subplots = len(to_plot.keys())

    if "axis" in kwargs.keys():
        del kwargs["axis"]

    fig, axes = aggregate_subplots(to_plot=to_plot,
                                   funct=boxplot,
                                   n_bars=n_bars,
                                   **kwargs)

    return fig, axes


def _dict2df(to_plot_dict: dict, meta_key: str) -> pd.DataFrame:
    """Transform a dictionary into a DataFrame for catplotting"""
    to_plot_df = []

    for range, values in to_plot_dict.items():
        range_grouped = []
        for ds in values:
            values_ds = values[ds]
            values_ds = values_ds.to_frame(name="values")
            values_ds["Dataset"] = ds
            values_ds[meta_key] = "\n[".join(range.split(" ["))
            range_grouped.append(values_ds)
        range_grouped = pd.concat(range_grouped, axis=0)
        to_plot_df.append(range_grouped)
    to_plot_df = pd.concat(to_plot_df, axis=0)

    return to_plot_df


def add_cat_info(to_plot: pd.DataFrame, metadata_name: str) -> pd.DataFrame:
    """Add info (N, median value) to metadata category labels"""
    groups = to_plot.groupby(metadata_name)["values"]  #
    counts = {}
    for name, group in groups:
        counts[name] = group[~group.index.duplicated(keep='first')].index.size

    to_plot[metadata_name] = to_plot[metadata_name].apply(
        lambda x: x + "\nN: {}".format(counts[x]))

    return to_plot


def bplot_catplot(to_plot,
                  y_axis,
                  metadata_name,
                  axis=None,
                  **kwargs) -> tuple:
    """
    Create individual plot with grouped boxplots by metadata value

    Parameters
    ----------
    to_plot: pd.Dataframe
        Seaborn-formatted dataframe
    y_axis: str
        Name of the x-axis
    metadata_name: str
        Name of the metadata type
    axis : matplotlib.axes.Axis, optional
        if provided, the function will create the plot on the specified axis
    """
    labels = None
    return_figax = False
    orient = "v"
    if axis is None:
        return_figax = True
        fig, axis = plt.subplots(1)
        orient = "h"

    if orient == "v":
        x = metadata_name
        y = "values"
    elif orient == "h":
        x = "values"
        y = metadata_name

    # add N points to the axis labels
    to_plot = add_cat_info(to_plot, metadata_name=metadata_name)

    box = sns.boxplot(
        x=x,
        y=y,
        hue="Dataset",
        data=to_plot.set_index(np.arange(to_plot.index.size)),
        palette="Set2",
        ax=axis,
        showfliers=False,
        orient=orient,
    )
    
    grouped = to_plot.groupby([metadata_name, "Dataset"])
    single_obs_data = grouped.filter(lambda x: len(x) == 1)

    # Only add points for single-observation groups
    if not single_obs_data.empty:
        num_patches = len(axis.patches)
    
        # Set size on a scale from 10 (when patches=5) to 4 (when patches=200 or more)
        if num_patches <= 5:
            point_size = 10
        elif num_patches >= 200:
            point_size = 7
        else:
            point_size = 10 - ((num_patches - 5) / (200 - 5)) * (10 - 7)
        sns.stripplot(
            x=x,
            y=y,
            hue="Dataset",
            data=single_obs_data.set_index(np.arange(single_obs_data.index.size)),
            palette="Set2",  # Same palette as boxplot
            ax=axis,
            size=point_size,         # Point size
            dodge=True,     # This aligns the points with their respective boxes
            jitter=False,   # Disable jitter to keep points centered
            orient=orient,
            legend=False    # Avoid duplicate legend
        )

    n_bars = to_plot["Dataset"].nunique()
    n_meta = to_plot[metadata_name].nunique()
    unit_height = 1
    unit_width = len(to_plot[metadata_name].unique())
    # needed for overlapping station names
    box.tick_params(labelsize=globals.tick_size)
    dims = [globals.boxplot_width * n_meta * 2, globals.boxplot_height]
    if metadata_name == 'network':
        dims = [
            globals.meta_network_boxplot_width +
            n_meta * globals.meta_network_width_scale_rate,
            globals.meta_network_boxplot_height +
            n_meta * globals.meta_network_boxplot_height_scale_factor
        ]
        if n_bars > 1:
            scaling_factor = 1 + 2 * (n_bars - 1) / 4
            dims[1] = dims[1] * scaling_factor
        
        # change y-labels to one line
        y_labels = [label.get_text() for label in box.get_yticklabels()]
        y_labels_fixed = [label.replace("\n", ", ") for label in y_labels]
        box.set_yticklabels(y_labels_fixed, fontsize=globals.tick_size)
    if orient == "v":
        axis.set(xlabel=None, ylabel=y_axis)
        axis.yaxis.grid(True)  # Hide the horizontal gridlines
        axis.xaxis.grid(False)  # Show the vertical gridlines

    if orient == "h":
        axis.set(ylabel=None, xlabel=y_axis)
        axis.yaxis.grid(False)  # Hide the horizontal gridlines
        axis.xaxis.grid(True)  # Show the vertical gridlines

    axis.set_axisbelow(True)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.legend(loc="best", fontsize="small")

    if return_figax:
        fig.set_figwidth(dims[0])
        fig.set_figheight(dims[1])

        return fig, axis

    else:
        axis.set(xlabel=None)
        axis.set(ylabel=None)


def boxplot_metadata(
    df: pd.DataFrame,
    metadata_values: pd.DataFrame,
    offset=0.02,
    ax_label=None,
    nbins=4,
    axis=None,
    plot_type: str = "catplot",
    meta_boxplot_min_samples=5,
    **bplot_kwargs,
) -> tuple:
    """
    Boxplots by metadata. The output plot depends on the metadata type:

    - "continuous"
    - "discrete"
    - "classes"

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with values for all variables (in metric)
    metadata_values : pd.DataFrame
        Dataframe containing the metadata values to use for the plot
    offset: float
        offset of watermark
    ax_label : str
        Name of the y axis - cannot be set globally
    nbins: int
        number pf bins to divide the plots in (only for continuous type of metadata, e.g. elevation)
    axis : matplotlib.axes.Axis, optional
        if provided, the function will create the plot on the specified axis
    plot_type : str, default is 'catplot'
        one of 'catplot' or 'multiplot', defines the type of plots for the 'classes' and 'continuous'
        metadata types
    meta_boxplot_min_samples: int, optional (default: 5)
        Minimum number of points in a bin to be plotted.
        If not enough points are available, the plot is not created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        the boxplot
    ax : matplotlib.axes.Axes
    labels : list
        list of class/ bins names
    """
    metric_label = "values"
    meta_key = metadata_values.columns[0]
    # sort data according to the metadata type
    metadata_type = globals.metadata[meta_key][2]

    bin_funct = bin_function_lut(metadata_type)
    to_plot = bin_funct(
        df=df,
        metadata_values=metadata_values,
        meta_key=meta_key,
        nbins=nbins,
        min_size=meta_boxplot_min_samples,
    )
    if to_plot is None:
        raise PlotterError(
            "There are too few points per metadata to generate the boxplots. You can set 'min_size'"
            "to a lower value to allow for smaller samples.")

    if isinstance(to_plot, dict):
        if plot_type == "catplot":
            to_plot = _dict2df(to_plot, meta_key)
            generate_plot = bplot_catplot
        elif plot_type == "multiplot":
            generate_plot = bplot_multiple

    elif isinstance(to_plot, pd.DataFrame):
        generate_plot = bplot_catplot

    out = generate_plot(
        to_plot=to_plot,
        y_axis=ax_label,
        metadata_name=meta_key,
        n_bars=len(df.columns),
        axis=axis,
        **bplot_kwargs,
    )

    if axis is None:
        fig, axes = out

        return fig, axes


def mapplot(
    df: pd.DataFrame,
    metric: str,
    ref_short: str,
    scl_short: Optional[str] = None,
    ref_grid_stepsize: Optional[float] = None,
    plot_extent: Optional[Tuple[float, float, float, float]] = None,
    colormap=None,
    projection: Optional[ccrs.Projection] = None,
    add_cbar: Optional[bool] = True,
    label: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = globals.map_figsize,
    dpi: Optional[int] = globals.dpi_min,
    diff_map: Optional[bool] = False,
    **style_kwargs: Dict
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
        Create an overview map from df using values as color. Plots a scatterplot for ISMN and an image plot for other
        input values.

        Parameters
        ----------
        df : pandas.Series
            values to be plotted. Generally from metric_df[Var]
        metric : str
            name of the metric for the plot
        ref_short : str
                short_name of the reference dataset (read from netCDF file)
        scl_short : str, default is None
            short_name of the scaling dataset (read from netCDF file).
            None if no scaling method is selected in validation.
        ref_grid_stepsize : float or None, optional (None by default)
                angular grid stepsize, needed only when ref_is_angular == False,
        plot_extent : tuple or None
                (x_min, x_max, y_min, y_max) in Data coordinates. The default is None.
        colormap :  Colormap, optional
                colormap to be used.
                If None, defaults to globals._colormaps.
        projection :  cartopy.crs, optional
                Projection to be used. If none, defaults to globals.map_projection.
                The default is None.
        add_cbar : bool, optional
                Add a colorbar. The default is True.
        label : str, optional
            Label of the y-axis, describing the metric. If None, a label is autogenerated from metadata.
            The default is None.
        figsize : tuple, optional
            Figure size in inches. The default is globals.map_figsize.
        dpi : int, optional
            Resolution for raster graphic output. The default is globals.dpi.
        diff_map : bool, default is False
            if True, a difference colormap is created
        **style_kwargs :
            Keyword arguments for plotter.style_map().

        Returns
        -------
        fig : matplotlib.figure.Figure
            the boxplot
        ax : matplotlib.axes.Axes
        """
    if not colormap:
        cmap = globals._colormaps[metric]
    else:
        cmap = colormap
    v_min, v_max = get_value_range(df, metric)
    # everything changes if the plot is a difference map
    if diff_map:
        v_min, v_max = get_value_range(df, metric=None, diff_map=True)
        cmap = globals._diff_colormaps[metric]

    if metric == 'status':
        df = _replace_status_values(df)
        labs = list(globals.status.values())
        cls = globals.get_status_colors().colors
        vals = sorted(list(set(df.values)))
        add_cbar = False

    # No need to mask ranged in the comparison plots
    else:
        # mask values outside range (e.g. for negative STDerr from TCA)
        if metric in globals._metric_mask_range.keys():
            mask_under, mask_over = globals._metric_mask_range[
                metric]  # get values from scratch to disregard quantiles
            cmap = copy.copy(cmap)
            if mask_under is not None:
                v_min = mask_under
                cmap.set_under("red")
            if mask_over is not None:
                v_max = mask_over
                cmap.set_over("red")

    # initialize plot
    fig, ax, cax = init_plot(figsize, dpi, add_cbar, projection)

    # scatter point or mapplot
    if ref_short in globals.scattered_datasets:  # scatter
        if not plot_extent:
            plot_extent = get_plot_extent(df)

        markersize = globals.markersize**2
        lat, lon, gpi = globals.index_names
        im = ax.scatter(df.index.get_level_values(lon),
                        df.index.get_level_values(lat),
                        c=df,
                        cmap=cmap,
                        s=markersize,
                        vmin=v_min,
                        vmax=v_max,
                        edgecolors='black',
                        linewidths=0.1,
                        zorder=2,
                        transform=globals.data_crs)
        if metric == 'status':
            ax.legend(handles=[
                Patch(facecolor=cls[x], label=labs[x])
                for x in range(len(globals.status)) if (x - 1) in vals
            ],
                      loc='lower center',
                      ncol=4)

    else:  # mapplot
        if not plot_extent:
            plot_extent = get_plot_extent(df,
                                          grid_stepsize=ref_grid_stepsize,
                                          grid=True)
        if isinstance(ref_grid_stepsize, np.ndarray):
            ref_grid_stepsize = ref_grid_stepsize[0]
        zz, zz_extent, origin = geotraj_to_geo2d(
            df, grid_stepsize=ref_grid_stepsize)  # prep values
        im = ax.imshow(zz,
                       cmap=cmap,
                       vmin=v_min,
                       vmax=v_max,
                       interpolation='nearest',
                       origin=origin,
                       extent=zz_extent,
                       transform=globals.data_crs,
                       zorder=2)

        if metric == 'status':
            ax.legend(handles=[
                Patch(facecolor=cls[x], label=labs[x])
                for x in range(len(globals.status)) if (x - 1) in vals
            ],
                      loc='lower center',
                      ncol=4)

    if add_cbar:  # colorbar
        _make_cbar(fig,
                   im,
                   cax,
                   ref_short,
                   metric,
                   label=label,
                   diff_map=diff_map,
                   scl_short=scl_short)
    style_map(ax, plot_extent, **style_kwargs)

    return fig, ax


def plot_spatial_extent(
    polys: dict,
    ref_points: bool = None,
    overlapping: bool = False,
    intersection_extent: tuple = None,
    reg_grid=False,
    grid_stepsize=None,
    **kwargs,
):
    """
    Plots the given Polygons and optionally the reference points on a map.

    Parameters
    ----------
    polys : dict
        dictionary with shape {name: shapely.geometry.Polygon}
    ref_points : 2D array of lon, lat for the reference points positions
    overlapping : bool, dafault is False.
        Whether the polygons have an overlap
    intersection_extent : tuple | None
        if given, corresponds to the extent of the intersection. Shape (minlon, maxlon, minlat, maxlat)
    reg_grid : bool, default is False,
        plotting oprion for regular grids (satellites)
    grid_stepsize:
    """
    fig, ax, cax = init_plot(figsize=globals.map_figsize, dpi=globals.dpi_min)
    legend_elements = []
    # plot polygons
    for n, items in enumerate(polys.items()):
        name, Pol = items
        if n == 0:
            union = Pol
        # get maximum extent
        union = union.union(Pol)
        style = {'color': 'powderblue', 'alpha': 0.4}
        # shade the union/intersection of the polygons
        if overlapping:
            x, y = Pol.exterior.xy
            if name == "selection":
                ax.fill(x, y, **style, zorder=5)
                continue
            ax.plot(x, y, label=name)
        # shade the areas individually
        else:
            if name == "selection":
                continue
            x, y = Pol.exterior.xy
            ax.fill(x, y, **style, zorder=6)
            ax.plot(x, y, label=name, zorder=6)
    # add reference points to the figure
    if ref_points is not None:
        if overlapping and intersection_extent is not None:
            minlon, maxlon, minlat, maxlat = intersection_extent
            mask = (ref_points[:, 0] >= minlon) & (ref_points[:, 0] <= maxlon) & \
                   (ref_points[:, 1] >= minlat) & (ref_points[:, 1] <= maxlat)
            selected = ref_points[mask]
            outside = ref_points[~mask]
        else:
            selected, outside = ref_points, np.array([])
        marker_styles = [
            {
                "marker": "o",
                "c": "turquoise",
                "s": 15
            },
            {
                "marker": "o",
                "c": "tomato",
                "s": 15
            },
        ]
        # mapplot with imshow for gridded (non-ISMN) references
        if reg_grid:
            plot_df = []
            for n, (point_set, style, name) in enumerate(
                    zip((selected, outside), marker_styles,
                        ("Selected reference validation points",
                         "Validation points outside selection"))):
                if point_set.size != 0:
                    point_set = point_set.transpose()
                    index = pd.MultiIndex.from_arrays(point_set,
                                                      names=('lon', 'lat'))
                    point_set = pd.Series(
                        data=n,
                        index=index,
                    )
                    plot_df.append(point_set)
                    # plot point to 'fake' legend entry
                    ax.scatter(0,
                               0,
                               label=name,
                               marker="s",
                               s=10,
                               c=style["c"])
                else:
                    continue
            plot_df = pd.concat(plot_df, axis=0)
            zz, zz_extent, origin = geotraj_to_geo2d(
                plot_df, grid_stepsize=grid_stepsize)
            cmap = mcol.LinearSegmentedColormap.from_list(
                'mycmap', ['turquoise', 'tomato'])
            im = ax.imshow(zz,
                           cmap=cmap,
                           origin=origin,
                           extent=zz_extent,
                           transform=globals.data_crs,
                           zorder=4)
        # scatterplot for ISMN reference
        else:
            for point_set, style, name in zip(
                (selected, outside), marker_styles,
                ("Selected reference validation points",
                 "Validation points outside selection")):
                if point_set.size != 0:
                    im = ax.scatter(point_set[:, 0],
                                    point_set[:, 1],
                                    edgecolors='black',
                                    linewidths=0.1,
                                    zorder=4,
                                    transform=globals.data_crs,
                                    **style,
                                    label=name)
                else:
                    continue
    # style plot
    make_watermark(fig, globals.watermark_pos, offset=0)
    title_style = {"fontsize": 12}
    ax.set_title("Spatial extent of the comparison", **title_style)
    # provide extent of plot
    d_lon = abs(union.bounds[0] - union.bounds[2]) * 1 / 8
    d_lat = abs(union.bounds[1] - union.bounds[3]) * 1 / 8
    plot_extent = (union.bounds[0] - d_lon, union.bounds[2] + d_lon,
                   union.bounds[1] - d_lat, union.bounds[3] + d_lat)
    grid_intervals = [1, 5, 10, 30]
    style_map(ax, plot_extent, grid_intervals=grid_intervals)
    # create legend
    plt.legend(loc='lower center',
               bbox_to_anchor=(0.5, -0.15),
               fontsize='x-small',
               framealpha=0.95,
               facecolor="white",
               edgecolor="white")


def _res2dpi_fraction(res, units):
    # converts a certain validation resolution to a 0-1 fraction
    # indicating the output quality
    # equivalent min/max ranges for km and degrees based on
    # available datasets, approximated
    res_range = {
        "km": [1, 36],
        "deg": [0.01, 0.33],
    }

    fraction = (res - min(res_range[units])) / (max(res_range[units]) -
                                                min(res_range[units]))

    return (1 - fraction)**2


def _extent2dpi_fraction(extent):
    # converts a certain validation extent to a 0-1 fraction
    # indicating the output quality
    max_extent = 360 * 110
    actual_extent = (extent[1] - extent[0]) * (extent[3] - extent[2])

    return actual_extent / max_extent


def output_dpi(res,
               units,
               extent,
               dpi_min=globals.dpi_min,
               dpi_max=globals.dpi_max) -> float:
    # get ouput dpi based on image extent and validation resolution
    # dpi = SQRT(extent_coeff^2 + res_coeff^2)
    dpi_vec = _extent2dpi_fraction(extent)**2 + _res2dpi_fraction(res,
                                                                  units)**2
    dpi_vec = np.sqrt(dpi_vec)
    dpi_fraction = dpi_vec / np.sqrt(2)

    dpi = dpi_min + (dpi_max - dpi_min) * dpi_fraction

    return float(dpi)


def average_non_additive(values: Union[pd.Series, np.array],
                         nobs: pd.Series) -> float:
    """
    Calculate the average of non-additive values, such as correlation
    scores, as recommended in:

    R. Alexander. A note on averaging correlations. Bulletin of the Psychonomic Society volume,
    1990.
    """
    # Try to get an array, unless already specified as np.array
    try:
        values = values.values

    except AttributeError:
        pass
    # Same for the nobs values
    try:
        nobs = nobs.values

    except AttributeError:
        pass

    # Transform to Fisher's z-scores
    z_scores = np.arctanh(values)
    # Remove the entries where there are NaNs
    mask = np.isfinite(values) & np.isfinite(nobs)
    z_scores = z_scores[mask]
    nobs = nobs[mask]

    # Get the number of points after droppin invalid
    k = len(z_scores)
    # Average taking the sample size into account
    mean = np.sum((nobs - 3) * z_scores) / (np.sum(nobs) - 3 * k)

    # Back transform the result
    return np.tanh(mean)


def scale_figure_for_network_metadata_plot(fig: "matplotlib.figure.Figure",
                                           ax: "matplotlib.axes.Axes",
                                           watermark_scale: float) -> Tuple:
    """
    Scales figure elements based on the number of patches.
    
    This function adjusts font sizes of various figure elements including title,
    tick labels, axis labels, and legend text. It also adjusts the layout and
    subplot parameters based on the number of patches in the axes.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to scale.
    ax : matplotlib.axes.Axes
        The axes object containing the patches to consider for scaling.
    watermark_scale : float
        The initial watermark scale factor that will be modified.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The scaled figure object.
    ax : matplotlib.axes.Axes
        The axes object, potentially modified.
    scale : float
        The modified scale value.
    """
    n_networks = len(ax.get_yticks())
    patches_count = len(ax.patches) # For case when multiple datasets are validated

    base_factor = globals.meta_network_base_font_size +  n_networks * globals.meta_network_font_scale_rate

    if patches_count > 50:
        # Scale linearly from 0 at 50 patches to 0.4 at 200 patches
        additional_scaling = 0.5 * (patches_count - 50) / (200 - 50)
        factor = base_factor * (1 + additional_scaling)
    else:
        factor = base_factor

    suptitle_text = fig._suptitle
    suptitle_text.set_fontsize(suptitle_text.get_fontsize() * factor)

    for tick in ax.get_xticklabels():
        tick.set_fontsize(tick.get_fontsize() * factor)

    for tick in ax.get_yticklabels():
        tick.set_fontsize(tick.get_fontsize() * factor)

    xlabel = ax.xaxis.label
    ylabel = ax.yaxis.label
    xlabel.set_fontsize(xlabel.get_fontsize() * factor)
    ylabel.set_fontsize(ylabel.get_fontsize() * factor)

    legend = ax.get_legend()
    for text in legend.get_texts():
        text.set_fontsize(text.get_fontsize() * factor)

    for patch in legend.get_patches():
        patch_size = 1.0 * (factor * 0.75)  # Scale factor based on n_networks
        patch.set_height(patch.get_height() * patch_size)

    fig.tight_layout(pad=2 * factor)
    new_bottom = fig.subplotpars.bottom * globals.meta_network_increase_padding_rate
    fig.subplots_adjust(bottom=new_bottom)

    # scale is to adjust size of Watermark, decreased here with increased number of networks.
    watermark_scale -= factor * 0.0116

    if watermark_scale < 0:
        watermark_scale = 0.05

    return fig, ax, watermark_scale


#$$
class ClusteredBoxPlot:
    """
    Class to create an empty figure object with one main axis and optionally three sub-axis. It is used to create a template for the clustered boxplot, which can then be filled with data.
    """

    def __init__(self,
                 anchor_list: Union[List[float], np.ndarray],
                 no_of_ds: int,
                 space_per_box_cluster: Optional[float] = 0.9,
                 rel_indiv_box_width: Optional[float] = 0.9):
        self.anchor_list = anchor_list
        self.no_of_ds = no_of_ds
        self.space_per_box_cluster = space_per_box_cluster
        self.rel_indiv_box_width = rel_indiv_box_width

        # xticklabel and legend label templates
        # self.xticklabel_template = "{tsw}:\n{dataset_name}\n({dataset_version})\nVariable: {variable_name} [{unit}]\n Median: {median:.3e}\n IQR: {iqr:.3e}\nN: {count}"
        self.xticklabel_template = "Median: {median:.3e}\n IQR: {iqr:.3e}\nN: {count}"
        self.label_template = "{dataset_name} ({dataset_version})\nVariable: {variable_name} [{unit}]"

    @staticmethod
    def centers_and_widths(
            anchor_list: Union[List[float], np.ndarray],
            no_of_ds: int,
            space_per_box_cluster: Optional[float] = 0.9,
            rel_indiv_box_width: Optional[float] = 0.9) -> List[CWContainer]:
        """
        Function to calculate the centers and widths of the boxes of a clustered boxplot. The function returns a list of tuples, each containing the center and width of a box in the clustered boxplot. The output can then be used as indices for creating the boxes a boxplot using `matplotlib.pyplot.boxplot()`

        Parameters
        ----------

        anchor_list: Union[List[float], np.ndarray]
            A list of floats representing the anchor points for each box cluster
        no_of_ds: int
            The number of datasets, i.e. the number of boxes in each cluster
        space_per_box_cluster: float
            The space each box cluster can occupy, 0.9 per default. This value should be <= 1 for a clustered boxplot to prevent overlap between neighboring clusters and boxes
        rel_indiv_box_width: float
            The relative width of the individual boxes in a cluster, 0.9 per default. This value should be <= 1 to prevent overlap between neighboring boxes

        Returns
        -------

        List[CWContainer]
            A list of CWContainer objects. Each dataset present has its own CWContainer object, each containing the centers and widths of the boxes in the clustered boxplot

        """

        b_lb_list = [
            -space_per_box_cluster / 2 + anchor for anchor in anchor_list
        ]  # list of lower bounds for each box cluster
        b_ub_list = [
            space_per_box_cluster / 2 + anchor for anchor in anchor_list
        ]  # list of upper bounds for each box cluster

        _centers = sorted([(b_ub - b_lb) / (no_of_ds + 1) + b_lb + i *
                           ((b_ub - b_lb) / (no_of_ds + 1))
                           for i in range(int(no_of_ds))
                           for b_lb, b_ub in zip(b_lb_list, b_ub_list)])
        _widths = [
            rel_indiv_box_width * (_centers[0] - b_lb_list[0])
            for _center in _centers
        ]

        return [
            CWContainer(name=f'ds_{ds}',
                        centers=_centers[ds::no_of_ds],
                        widths=_widths[ds::no_of_ds])
            for ds in range(int(no_of_ds))
        ]

    @staticmethod
    def figure_template(incl_median_iqr_n_axs: Optional[bool] = False,
                        **fig_kwargs) -> ClusteredBoxPlotContainer:
        """
        Function to create a figure template for e.g. a clustered boxplot. The function returns a \
        ClusteredBoxPlotContainer object, which contains the figure and the subplots for the boxplot as well as \
        optionally the median, IQR and N values. The layout is as follows: the axes are arranged in a 2x1 grid, \
        with the boxplot in the upper subplot and the median, IQR and N values in the lower subplot. \
        The lower subplot is further divided into three subplots, one for each value.

        Parameters
        ----------
        incl_median_iqr_n_axs: Optional[bool]
            If True, creates three subplots with median, IQR and N values for each box. If False, only the boxplot is \
                created. Default is False
        fig_kwargs: dict
            Keyword arguments for the figure

        Returns
        -------
        ClusteredBoxPlotContainer
            A ClusteredBoxPlotContainer object containing the figure and the subplots for the boxplot, median, \
                IQR and N values
        """

        if 'figsize' in fig_kwargs:
            _fig = plt.figure(figsize=fig_kwargs['figsize'])
        else:
            _fig = plt.figure(figsize=(15, 10.5))

        if not incl_median_iqr_n_axs:
            ax_box = _fig.add_subplot(111)
            ax_median, ax_iqr, ax_n = None, None, None

        if incl_median_iqr_n_axs:
            # Create a main gridspec for ax_box and subplots below
            gs_main = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.2)

            # Subgridspec for ax_box and ax_median (top subplot)
            gs_top = gridspec.GridSpecFromSubplotSpec(1,
                                                      1,
                                                      subplot_spec=gs_main[0])

            # Subgridspec for ax_iqr and ax_n (bottom subplots)
            gs_bottom = gridspec.GridSpecFromSubplotSpec(
                3,
                1,
                height_ratios=[1, 1, 1],
                subplot_spec=gs_main[1],
                hspace=0)
            ax_box = plt.subplot(gs_top[0])
            ax_median = plt.subplot(gs_bottom[0], sharex=ax_box)
            ax_iqr = plt.subplot(gs_bottom[1], sharex=ax_box)
            ax_n = plt.subplot(gs_bottom[2], sharex=ax_box)

        for _ax in [ax_box, ax_median, ax_iqr, ax_n]:
            try:
                _ax.tick_params(labelsize=globals.tick_size)
                _ax.spines['right'].set_visible(False)
                _ax.spines['top'].set_visible(False)
            except AttributeError:
                pass

        add_logo_to_figure(
            fig=_fig,
            logo_path=globals.watermark_logo_pth,
            position=globals.watermark_logo_position,
            offset=globals.watermark_logo_offset_comp_plots,
            scale=globals.watermark_logo_scale,
        )

        return ClusteredBoxPlotContainer(fig=_fig,
                                         ax_box=ax_box,
                                         ax_median=ax_median,
                                         ax_iqr=ax_iqr,
                                         ax_n=ax_n)
