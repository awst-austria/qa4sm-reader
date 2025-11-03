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
from scipy.ndimage import rotate
from scipy.spatial.distance import pdist, squareform
import os.path

from typing import Union, List, Tuple, Dict, Optional, Any
import copy

import seaborn as sns
import matplotlib
import matplotlib.axes
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch, PathPatch, Rectangle
from qa4sm_reader.colors import get_color_for, get_palette_for

from cartopy import config as cconfig
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs

from pygeogrids.grids import BasicGrid, genreg_grid
from shapely.geometry import Polygon, Point

import warnings
import os
from collections import namedtuple

import textwrap

# Change of standard matplotlib parameters
matplotlib.rcParams['legend.framealpha'] = globals.legend_alpha
plt.rcParams['hatch.linewidth'] = globals.hatch_linewidth
# Change of standard seaborn boxplot parameters through monkeypatching
_old_boxplot = sns.boxplot

def custom_boxplot(*args, **kwargs):
    defaults = dict(
        boxprops=dict(edgecolor=globals.boxplot_edgecolor, linewidth=globals.boxplot_edgewidth),
        whiskerprops=dict(color=globals.boxplot_edgecolor, linewidth=globals.boxplot_edgewidth),
        capprops=dict(color=globals.boxplot_edgecolor, linewidth=globals.boxplot_edgewidth),
        medianprops=dict(color=globals.boxplot_edgecolor, linewidth=globals.boxplot_edgewidth),
    )
    for k, v in defaults.items():
        if k in kwargs:
            defaults[k].update(kwargs.pop(k))
    return _old_boxplot(*args, **kwargs, **defaults)

sns.boxplot = custom_boxplot

cconfig['data_dir'] = os.path.join(os.path.dirname(__file__), 'cartopy')

def wrapped_text(fig, text, width, fontsize) -> str:
    """
    Wrap a long string of text to fit into a given figure width.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object in which the text will be drawn.
    text : str
        The text to wrap.
    width : float
        The available width in pixels for the text.
    fontsize : int
        The font size in points used for estimating text width.

    Returns
    -------
    wrapped : str
        The text wrapped into multiple lines, separated by '\n'.
    """
    # Validate fontsize
    fontsize=float(fontsize)
    if not np.isfinite(fontsize) or fontsize <= 0:
        warnings.warn(f"Invalid fontsize {fontsize}, using fallback value of 10")
        fontsize = 10
    
    # Validate width
    if not np.isfinite(width) or width <= 0:
        warnings.warn(f"Invalid width {width}, returning unwrapped text")
        return text
    
    sample = "This is a very long text that should automatically wrap into multiple lines depending on the figure width"
    
    try:
        if not fig.axes:
            ax = fig.add_subplot(111)
            ax.set_axis_off()
        else:
            ax = fig.axes[0]
        example_text = ax.text(0.5,0.5,sample, fontsize=fontsize)
        renderer = fig.canvas.get_renderer()
        
        try:
            text_extent = example_text.get_window_extent(renderer=renderer)
            char_width_px = text_extent.width / len(sample)
        except (RuntimeError, ValueError) as e:
            warnings.warn(f"Could not measure text extent: {e}. Using fallback character width estimation.")
            # Fallback: approximate character width based on fontsize
            # Typical monospace character is about 0.6 * fontsize in pixels at 72 dpi
            char_width_px = fontsize * fig.dpi / 72 * 0.6
        
        example_text.set_text("")
        
        # Validate char_width_px
        if not np.isfinite(char_width_px) or char_width_px <= 0:
            warnings.warn("Invalid character width calculated, using fallback")
            char_width_px = fontsize * fig.dpi / 72 * 0.6
        
        # wrap text
        max_chars = int(width / char_width_px * 1)
        
        # Ensure max_chars is reasonable
        if max_chars < 10:
            max_chars = 10  # minimum sensible line length
        
        wrapped = "\n".join(textwrap.wrap(text, max_chars))
        return wrapped
        
    except Exception as e:
        warnings.warn(f"Text wrapping failed: {e}. Returning original text.")
        return text

def best_legend_pos_exclude_list(ax, forbidden_locs= globals.leg_loc_forbidden):
    """
    Find the best legend position, excluding a list of positions.
    
    Parameters:
        ax : matplotlib.axes.Axes
        forbidden_locs : list of str or numbers, e.g. ["lower right", 2]
    
    Returns:
        best_loc_str : string of the best location
    """
    # standard Matplotlib positions
    locs = globals.leg_loc_dict
    
    # resolve forbidden positions to numbers
    forbidden_nums = set()
    for loc in forbidden_locs:
        if isinstance(loc, str):
            num = locs.get(loc)
            if num is not None:
                forbidden_nums.add(num)
        else:
            forbidden_nums.add(loc)
    
    # candidate positions
    candidate_locs = [loc for loc in locs.values() if loc not in forbidden_nums]
    
    fig = ax.figure
    
    min_overlap = float("inf")
    best_loc = candidate_locs[0]
    
    # evaluate overlap for each candidate
    leg = ax.get_legend()
    if not leg:
        leg = ax.legend()
    for loc in candidate_locs:
        leg.set_loc(loc=loc)
        fig.canvas.draw()
        
        bbox_legend = leg.get_window_extent()
        xdata = [line.get_xdata() for line in ax.get_lines()]
        ydata = [line.get_ydata() for line in ax.get_lines()]
        
        overlap = 0
        for xd, yd in zip(xdata, ydata):
            for x, y in zip(xd, yd):
                xpix, ypix = ax.transData.transform((x, y))
                if bbox_legend.contains(xpix, ypix):
                    overlap += 1
        
        if overlap < min_overlap:
            min_overlap = overlap
            best_loc = loc
    
    # convert numeric back to string
    best_loc_str = {v:k for k,v in locs.items()}[best_loc]
    return best_loc_str

def pixel_distance_nearest(ax, sc, min_px=10) -> float:
    """
    Compute the minimum center-to-center pixel distance between markers in a scatter plot,
    ignoring distances smaller than `min_px` (e.g., duplicates or rounding artifacts).

    Parameters
    ----------
    ax : matplotlib Axes or Cartopy GeoAxes
        The axes containing the scatter.
    sc : PathCollection
        Scatter object returned by ax.scatter.
    min_px : float
        Minimum distance in pixels to consider; distances smaller than this are ignored.

    Returns
    -------
    float
        Minimum distance in pixels between markers (ignoring distances < min_px).
    """
    fig = ax.figure; fig.canvas.draw()
    T = sc.get_offset_transform()
    xy = T.transform(sc.get_offsets())
    D = squareform(pdist(xy))
    np.fill_diagonal(D, np.inf)
    D[D < min_px] = np.inf
    i, j = np.unravel_index(np.argmin(D), D.shape)
    return D[i, j]

def non_overlapping_markersize(ax, scatter):
    """
    Compute a scatter marker size (points^2) so markers just touch without overlapping,
    using the actual drawn positions on the figure.

    Parameters
    ----------
    ax : matplotlib Axes or Cartopy GeoAxes
    scatter : PathCollection
        Scatter object returned by ax.scatter.

    Returns
    -------
    float
        Marker size to use in scatter (s parameter).
    """
    dist = pixel_distance_nearest(ax, scatter)
    margin = 1 # pixel margin to avoid edges of overlapping
    dp = dist - margin

    # convert pixel radius → points² (as required by scatter s)
    # 1 point = fig.dpi/72 pixels
    radius_points = dp * 72.0 / ax.figure.dpi
    size = (globals.min_markersize if (radius_points**2) < globals.min_markersize else 
            globals.max_markersize if (radius_points**2) > globals.max_markersize else 
            (radius_points**2))
    return size

def _float_gcd(a, b, atol=1e-04):
    "Greatest common divisor (=groesster gemeinsamer teiler)"
    while abs(b) > atol:
        a, b = b, a % b
    return a


def _get_grid(a):
    "Find the stepsize of the grid behind a and return the parameters for that grid axis."
    a = np.unique(a)  # get unique values and sort
    das = np.unique(np.diff(a))  # get unique stepsizes and sort
    da = das[0]  # get smallest stepsize
    dal = []
    for d in das[1:]:  # make sure, all stepsizes are multiple of da
        da = _float_gcd(d, da)
        dal.append(da)
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

    # set map extent to have aspect from at least 1/q or q/1
    width = extent[1]-extent[0]
    height = extent[3]-extent[2]
    q = 4
    if width < height/q: #enlargens width to be at least height/q
        extent[0] = (extent[1]+extent[0])/2 - height/(q*2)
        extent[1] = (extent[1]+extent[0])/2 + height/(q*2)

    elif height < width/q: #enlargens height to be at least width/q
        extent[2] = (extent[2]+extent[3])/2 - width/(q*2)
        extent[3] = (extent[2]+extent[3])/2 + width/(q*2)

    return extent

def init_plot(figsize,
              dpi=globals.dpi_min,
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

    ax_main = fig.add_axes([globals.map_ax_left, 
                            globals.map_ax_bottom, 
                            globals.map_ax_width, 
                            globals.map_ax_height], 
                            projection=projection)

    return fig, ax_main

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
    add_topo=False,
    add_coastline=True,
    add_land=True,
    add_water=True,
    add_borders=True,
    add_us_states=False,
    grid_intervals=globals.grid_intervals,
    grid_tick_size=None,
):
    """Parameters to style the mapplot"""
    ax.set_extent(plot_extent, crs=globals.data_crs)
    ax.spines["geo"].set_linewidth(ax.spines["top"].get_linewidth())
    min_extent = min ([abs(plot_extent[0]-plot_extent[1]), abs(plot_extent[2]-plot_extent[3])])
    map_resolution = (globals.naturalearth_resolution[0] if min_extent <= globals.resolution_th[0] else 
                      globals.naturalearth_resolution[1] if min_extent <= globals.resolution_th[1] else 
                      globals.naturalearth_resolution[2])

    if add_grid:
        try:
            # determine approximate grid interval
            grid_interval = max(plot_extent[1] - plot_extent[0],
                                plot_extent[3] - plot_extent[2]) / globals.min_gridlines

            # select closest available interval
            if grid_interval <= min(grid_intervals):
                raise RuntimeError("No suitable grid interval")
            grid_interval = min(grid_intervals, key=lambda x: abs(x - grid_interval))

            # compute tick positions within plot extent
            xticks = np.arange(-180, 180.001, grid_interval)
            yticks = np.arange(-90, 90.001, grid_interval)
            xticks = xticks[(xticks >= plot_extent[0]) & (xticks <= plot_extent[1])]
            yticks = yticks[(yticks >= plot_extent[2]) & (yticks <= plot_extent[3])]

            # single gridlines call
            gl = ax.gridlines(crs=globals.data_crs,
                            draw_labels=True,
                            linewidth=0.5,
                            color='grey',
                            linestyle='--',
                            zorder=3)
            gl.xlocator = mticker.FixedLocator(xticks)
            gl.ylocator = mticker.FixedLocator(yticks)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.top_labels = False
            gl.right_labels = False

            # set label fontsize
            gl.xlabel_style = {'size': globals.fontsize_ticklabel}
            gl.ylabel_style = {'size': globals.fontsize_ticklabel}

        except RuntimeError as e:
            print("Gridlines or labels not plotted.\n" + str(e))

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
                                            facecolor=globals.map_land_color)
        ax.add_feature(land, zorder=1)
    if add_water:
        ocean = cfeature.NaturalEarthFeature(category='physical', 
                                             name='ocean', 
                                             scale=map_resolution, 
                                             facecolor=globals.map_water_color)
        
        ax.add_feature(ocean)
    if add_borders:
        borders = cfeature.NaturalEarthFeature('cultural',
                                               'admin_0_countries',
                                               map_resolution,
                                               edgecolor='black',
                                               facecolor='none')
        ax.add_feature(borders, linewidth=0.5, zorder=3)
    if add_us_states:
        states = ax.add_feature(cfeature.STATES, linewidth=0.1, zorder=3)

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
    pad = globals.logo_pad
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

def add_logo_in_bg_front(
        fig: matplotlib.figure.Figure,
        logo_path: Optional[str] = globals.logo_pth,
        position: Optional[str] = globals.logo_position,
        size: Optional[float] = globals.logo_size,     # points, like fontsize
        alpha: float = globals.logo_alpha,  # transparency
        rotation: float = globals.logo_rotation, # degrees
    ) -> None:
    """
    Overlay a logo centered on every axis in the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to decorate.
    logo_path : str
        Path to logo image file.
    size : float
        Logo height in points (like fontsize).
    alpha : float
        Transparency of the logo [0-1].
    rotation : float
        Rotation angle in degrees.
    """
    if not os.path.exists(logo_path):
        raise FileNotFoundError(f"Logo not found at '{logo_path}'")

    im = mpimg.imread(logo_path)
    h, w, _ = im.shape
    aspect = w / h
    im = rotate(im, rotation, reshape=True)
    rotrad = rotation*np.pi*2/360

    dpi = fig.dpi
    wm_h = abs(np.sin(rotrad)*aspect*size)+abs(np.cos(rotrad)*size)
    wm_w = abs(np.sin(rotrad)*size)+abs(np.cos(rotrad)*aspect*size)
    logo_height_px = wm_h * dpi / 72.0
    logo_width_px = wm_w * dpi / 72.0

    # convert to figure coordinates
    
    logo_height_fig = logo_height_px / (fig.get_figheight() * dpi)
    logo_width_fig  = logo_width_px  / (fig.get_figwidth()  * dpi)

    for ax in fig.get_axes():
        # axis position in figure coords
        bbox = ax.get_position()
        if "bg" in position and not hasattr(ax, 'projection'):
            ax.set_facecolor("none")
        
        if globals.n_logo > 1: # Automatically adjust multiple logos
            s = 1
            for i in range(globals.n_logo):
                j = i % globals.n_col_logo
                left = bbox.x0 + globals.logo_pad*bbox.width + \
                    j * ((bbox.width - logo_width_fig- 2 * globals.logo_pad * bbox.width) / \
                         (globals.n_col_logo - 1))
                bottom = bbox.y0 + globals.logo_pad * bbox.height + \
                    i * ((bbox.height - logo_height_fig - 2 * globals.logo_pad * bbox.height)/ \
                         (globals.n_logo-1))
                ax_logo = fig.add_axes([
                left,
                bottom,
                logo_width_fig,
                logo_height_fig])
                s += 1

                ax_logo.imshow(im, alpha=alpha)
                ax_logo.axis("off")

                if "bg" in position and not hasattr(ax, 'projection'):
                    ax_logo.set_zorder(-1)
                
        elif globals.n_logo == 1: # Only one logo
            # logo axes
            # Lower Left Corner horizontal position
            if any(sub in ["right"] for sub in position.split("_")):
                left = bbox.x0 + bbox.width - globals.logo_pad - logo_width_fig
            elif any(sub in ["left"] for sub in position.split("_")):
                left = bbox.x0 + globals.logo_pad
            else: # center or not specified
                left = bbox.x0 + bbox.width/2 - logo_width_fig/2

            # Lower left corner vertical position
            if any(sub in ["upper"] for sub in position.split("_")):
                bottom = bbox.y0 + bbox.height - globals.logo_pad - logo_height_fig
            elif any(sub in ["lower"] for sub in position.split("_")):
                bottom = bbox.y0 + globals.logo_pad
            else: # center or not specified
                bottom = bbox.x0 + bbox.height/2 - logo_height_fig/2            

            ax_logo = fig.add_axes([
                left,
                bottom,
                logo_width_fig,
                logo_height_fig
            ])

            ax_logo.imshow(im, alpha=alpha)
            ax_logo.axis("off")

            if "bg" in position and not hasattr(ax, 'projection'):
                ax_logo.set_zorder(-1)
        else:
            break

        if hasattr(ax, 'projection'):
            break

def add_logo_to_figure(
        fig: matplotlib.figure.Figure,
        logo_path: Optional[str] = globals.logo_pth,
        position: Optional[str] = globals.logo_position,
        offset: Optional[Union[Tuple, Offset]] = (0., -0.05),
        y_pad: Optional[float] = 14,
        size: Optional[float] = globals.logo_size) -> None:
    """
    Add a logo to an existing figure.
    A size of 12 would be the same as fontsize 12.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the logo to. Must have at least one axis.
    logo_path : Optional[str]
        Path to the logo image. If None, use matplotlib's grace_hopper.png.
    position : Optional[str]
        'lower_left', 'lower_center', 'lower_right', 'upper_left', 'upper_center', 'upper_right'.
    offset : Optional[Tuple | Offset]
        Offset (x, y) in figure coordinates.
    y_pad : Optional[float]
        Padding between figure and logo (in fontsize).
    size : Optional[float]
        Logo height (in fontsize).
    """
    # positioning possibilities
    rel_to_plot = ["front", "bg", "side"]
    va_l = ["lower", "center", "upper"]
    ha_l = ["left", "center", "right"]
    if any(sub not in rel_to_plot + va_l + ha_l for sub in position.split("_")):
        warnings.warn(f"Position not implemented has to only include [{rel_to_plot+va_l+ha_l}]. Resorting to fallback: 'front_lower_right'")
        position = 'front_lower_right'
    if any(sub in ["bg", "front"] for sub in position.split("_")):
        add_logo_in_bg_front(fig, logo_path, position=position, size=size)
        return
    if not fig.get_axes():
        warnings.warn("No axes found in the figure. Creating a new one.")
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[-1] #-1 to always take the last axis

    if not os.path.exists(logo_path):
        warnings.warn(f"No logo found at: '{logo_path}'. Skipping logo addition.")
        return

    with cbook.get_sample_data(logo_path) as file:
        im = mpimg.imread(file)

    dpi = fig.dpi
    logo_height_px = size * dpi / 72.0

    h, w, _ = im.shape

    # scale separately in x and y
    logo_height_fig = logo_height_px / (fig.get_figheight() * dpi)
    logo_width_px   = w * logo_height_px / h
    logo_width_fig  = logo_width_px / (fig.get_figwidth() * dpi)
    y_pad_fig = (y_pad * dpi / 72) / (fig.get_figheight() * dpi)

    if not isinstance(offset, Offset):
        offset = Offset(*offset)

    fig_trans = fig.transFigure.inverted()

    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer=renderer)
    bbox_fig = bbox.transformed(fig_trans)

    # If Figure has super positioned titles/labels include those in bounding box
    extras = []
    if fig._suptitle is not None:
        extras.append(fig._suptitle.get_window_extent(renderer).transformed(fig.transFigure.inverted()))
    if hasattr(fig, "_supxlabel") and fig._supxlabel is not None:
        extras.append(fig._supxlabel.get_window_extent(renderer).transformed(fig.transFigure.inverted()))
    if hasattr(fig, "_supylabel") and fig._supylabel is not None:
        extras.append(fig._supylabel.get_window_extent(renderer).transformed(fig.transFigure.inverted()))

    for eb in extras:
        bbox_fig = matplotlib.transforms.Bbox.union([bbox_fig, eb])
    
    if 'lower' in position:
        bottom = bbox_fig.y0 - logo_height_fig - y_pad_fig
    elif 'upper' in position:
        bottom = bbox_fig.y1 + y_pad_fig
    else:
        bottom = 0 + offset.y  # fallback

    if 'left' in position:
        left = 0 + offset.x
    elif 'center' in position:
        left = 0.5 - logo_width_fig/2 + offset.x
    elif 'right' in position:
        left = 1 - logo_width_fig + offset.x
    
    # Add logo axis
    ax_logo = fig.add_axes([left, bottom, logo_width_fig, logo_height_fig])
    ax_logo.imshow(im)
    ax_logo.axis("off")


def _make_cbar(fig,
               ax,
               im,
               ref_short: str,
               metric: str,
               label=None,
               diff_map=False,
               scl_short=None,
               wrap_text=True):
    """
    Make colorbar to use in plots

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        figure of plot
    im: AxesImage
        from method Axes.imshow()
    ax: axes.SubplotBase
        from fig.add_axes
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

    if im is None or not hasattr(im, "get_array") or im.get_array() is None:
        warnings.warn("Skipping colorbar: invalid or empty image handle")
        return fig, im, None
    
    if label is None:
        label = globals._metric_name[metric]

    extend = get_extend_cbar(metric)
    if diff_map:
        extend = "both"

    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        fig.canvas.draw()  # guarantees renderer exists
    except Exception as e:
        warnings.warn(f"Couldn't draw figure for initializing renderer: {e}")
        pass
    
    bbox = ax.get_position()

    labels = []
    min_fontsize = 1.0  # minimum valid fontsize
    default_fontsize = max(globals.fontsize_ticklabel, min_fontsize)
    
    for lbl in ax.get_xticklabels():
        text = lbl.get_text()
        if not text:
            continue  # skip empty labels
        
        lbl.set_visible(True)  # ensure label is visible
        
        # Force valid font size BEFORE any rendering operation
        try:
            fs = lbl.get_fontsize()
            if not np.isfinite(fs) or fs < min_fontsize:
                lbl.set_fontsize(default_fontsize)
        except Exception:
            lbl.set_fontsize(default_fontsize)
        
        # Optionally skip labels off-axis
        try:
            x, y = lbl.get_position()
            if not (ax.get_xlim()[0] <= x <= ax.get_xlim()[1]):
                continue
        except Exception:
            pass  # include label if position check fails
        
        labels.append(lbl)
    
    if not labels:
        warnings.warn("No tick labels found for colorbar placement — using fallback positioning.")
        # Use axis position as fallback
        pad = 0.02  # 2% of figure height
        cax = fig.add_axes([bbox.x0, bbox.y0 - globals.cax_width - pad, 
                           bbox.width, globals.cax_width])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
        
        # Set label with validation
        try:
            fontsize = globals.fontsize_label
            cbar.set_label(label, fontsize=fontsize)
        except Exception as e:
            warnings.warn(f"Could not set colorbar label: {e}")
        
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor('black')
        cbar.ax.tick_params(width=0.6, labelsize=default_fontsize)
        return fig, im, cax
    
    # Try to get renderer, with fallback
    try:
        renderer = fig.canvas.get_renderer()
    except Exception as e:
        warnings.warn(f"Could not get renderer: {e}. Using fallback colorbar positioning.")
        pad = 0.02
        cax = fig.add_axes([bbox.x0, bbox.y0 - globals.cax_width - pad, 
                           bbox.width, globals.cax_width])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
        
        try:
            fontsize = globals.fontsize_label
            cbar.set_label(label, fontsize=fontsize)
        except Exception:
            pass
        
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor('black')
        cbar.ax.tick_params(width=0.6, labelsize=default_fontsize)
        return fig, im, cax
    
    # Get bounding boxes with individual error handling
    valid_bboxes = []
    for lbl in labels:
        try:
            # Double-check fontsize right before rendering
            fs = lbl.get_fontsize()
            if not np.isfinite(fs) or fs < min_fontsize:
                lbl.set_fontsize(default_fontsize)
            
            bbox_lbl = lbl.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            
            # Validate bbox
            if (np.isfinite(bbox_lbl.y0) and np.isfinite(bbox_lbl.y1) and 
                np.isfinite(bbox_lbl.x0) and np.isfinite(bbox_lbl.x1)):
                valid_bboxes.append(bbox_lbl)
        except (RuntimeError, ValueError, AttributeError) as e:
            # Skip this label silently - we already warned about invalid labels
            continue
    
    if not valid_bboxes:
        warnings.warn("No valid tick label extents — using fallback padding")
        pad = 5 / fig.dpi if fig.dpi > 0 else 0.02  # 5 pixels or 2% fallback
        valid_bboxes = [ax.get_position()]  # fallback
    
    # Calculate pad safely
    try:
        min_y1 = min([i.y1 for i in valid_bboxes])
        min_y0 = min([i.y0 for i in valid_bboxes])
        pad = bbox.y0 - min_y1
        
        # Validate pad
        if not np.isfinite(pad) or pad < 0:
            pad = 5 / fig.dpi if fig.dpi > 0 else 0.02
    except Exception:
        pad = 5 / fig.dpi if fig.dpi > 0 else 0.02
        min_y0 = bbox.y0 - 0.05  # fallback
    
    # Create colorbar axes
    cax = fig.add_axes([bbox.x0, min_y0 - globals.cax_width - pad, 
                        bbox.width, globals.cax_width])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend)
    
    # Set label with full error handling
    try:
        fontsize = globals.fontsize_label
        if not np.isfinite(fontsize) or fontsize <= 0:
            fontsize = 10
            warnings.warn(f"Invalid globals.fontsize_label, using {fontsize}")
        
        cax_pos = cax.get_position()
        label_width = fig.get_figwidth() * (cax_pos.x1 - cax_pos.x0) * fig.dpi
        
        if not np.isfinite(label_width) or label_width <= 0:
            wrapped_label = label
        else:
            if wrap_text:
                wrapped_label = wrapped_text(fig, label, label_width, fontsize)
            else:
                wrapped_label=label
        
        cbar.set_label(wrapped_label, fontsize=fontsize)
    except Exception as e:
        warnings.warn(f"Failed to set colorbar label: {e}")
        try:
            cbar.set_label(label, fontsize=10)
        except Exception:
            pass  # Give up on label if even simple setting fails
    
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(width=0.6, labelsize=default_fontsize)
    
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
        ax.annotate("Mean CI\nRange:\n {:.2g}".format(ci_range) if ci_range>0.01 else "Mean CI\nRange:\n <0.01",
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
        patch.set(color=globals.boxplot_edgecolor, facecolor=facecolor, linewidth=globals.boxplot_edgewidth, alpha=1)
        median.set(color=globals.boxplot_edgecolor, linewidth=globals.boxplot_edgewidth)
    for (whis, caps) in zip(box_dict["whiskers"], box_dict["caps"]):
        whis.set(color=globals.boxplot_edgecolor, linewidth=globals.boxplot_edgewidth)
        caps.set(color=globals.boxplot_edgecolor, linewidth=globals.boxplot_edgewidth)


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

def capsizing(ax, orient="v", factor=globals.cap_factor, iterative=True, n_lines=None):
    """Deals with adjusting the capsize of the boxplot whiskers to factor * boxwidth.
        Parameters
        ----------
        ax : matplotlib.axes object
            ax containing new lines of boxplot
        n_lines : int, optional
            number of preexisting lines before drawing the boxplot. 
            Only used when function called in loop.
        orient : str, optional
            which orientation does the boxplot have
        factor : float, optional
            factor of capwidth to boxwidth
        iterative : bool, optional
            determines if function gets called in loop or should loop 
            through boxplots by itself
    """
    if iterative:
        new_lines = ax.lines[n_lines:]
        cap1, cap2 = new_lines[2], new_lines[3]
        if orient == "v":
            dist = new_lines[4].get_xdata()[1] - new_lines[4].get_xdata()[0]
            center = new_lines[4].get_xdata().mean()
            cap1.set_xdata([center - dist*factor/2, center + dist*factor/2])
            cap2.set_xdata([center - dist*factor/2, center + dist*factor/2])
        elif orient == "h":
            dist = new_lines[4].get_ydata()[1] - new_lines[4].get_ydata()[0]
            center = new_lines[4].get_ydata().mean()
            cap1.set_ydata([center - dist*factor/2, center + dist*factor/2])
            cap2.set_ydata([center - dist*factor/2, center + dist*factor/2])
    elif not iterative:
        for i in range(int(len(ax.lines)/5)): # 5 lines per boxplot
            box_lines = ax.lines[i*5:(i+1)*5] # lines of each boxplot
            cap1, cap2 = box_lines[2], box_lines[3]
            if orient == "v":
                dist = box_lines[4].get_xdata()[1] - box_lines[4].get_xdata()[0]
                center = box_lines[4].get_xdata().mean()
                cap1.set_xdata([center - dist*factor/2, center + dist*factor/2])
                cap2.set_xdata([center - dist*factor/2, center + dist*factor/2])
            elif orient == "h":
                dist = box_lines[4].get_ydata()[1] - box_lines[4].get_ydata()[0]
                center = box_lines[4].get_ydata().mean()
                cap1.set_ydata([center - dist*factor/2, center + dist*factor/2])
                cap2.set_ydata([center - dist*factor/2, center + dist*factor/2])

def get_box_bbox_data(ax, box):
    """
    Return (x0, y0, x1, y1) bounding box of a seaborn/matplotlib
    box (PathPatch) in DATA coordinates.
    """
    # Vertices of the PathPatch (still in its own normalized coords)
    verts = box.get_path().vertices

    # First map through the box's transform (usually to display coords)
    verts_disp = box.get_transform().transform(verts)

    # Then map from display coords back to data coords
    verts_data = ax.transData.inverted().transform(verts_disp)

    # Bounding box corners in data coordinates
    x0, y0 = np.min(verts_data, axis=0)
    x1, y1 = np.max(verts_data, axis=0)
    return x0, y0, x1, y1

def triangle_hatching(ax, box, dist=0.5, direction="up", zorder=-1, linewidth=1, color="k", alpha=1):
    """
    Draw triangular hatching inside a box patch using a LineCollection (fast).

    This function overlays evenly spaced triangular lines across the bounding box 
    of the given `box` patch. Triangles can point in four directions: 'up', 'down', 
    'left', or 'right'.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which to draw.
    box : matplotlib.patches.Patch
        A rectangular patch (usually a PathPatch from a seaborn boxplot).
    dist : float, default=0.5
        Vertical (or horizontal) spacing between adjacent triangles in data units.
    direction : {"up", "down", "left", "right"}, default="up"
        Orientation of the triangles:
        - "up" : triangles point upwards
        - "down" : triangles point downwards
        - "left" : triangles point leftwards
        - "right" : triangles point rightwards
    zorder : int, default=-1
        Drawing order of the hatch lines relative to other artists.
    linewidth : float, default=1
        Width of the hatch lines.
    color : str or tuple, default="k"
        Color of the hatch lines.
    alpha : float, default=1
        Opacity of the hatch lines (0=transparent, 1=opaque).

    Notes
    -----
    - Uses a LineCollection to add all lines in one artist for speed.
    - Automatically handles truncated triangles at the edges.
    - Works with boxes from seaborn/matplotlib (PathPatch or Rectangle).
    """

    # Get bounding box in data coordinates
    x0, y0, x1, y1 = get_box_bbox_data(ax, box)
    lines = []

    if direction in ("up", "down"):
        y = y1 - y0
        for i in range(int(y // dist) + 1):
            if direction == "up":
                base_y = i * dist + y0
                if base_y + dist <= y1:  # full triangle
                    lines.append([(x0, base_y), ((x0 + x1)/2, base_y + dist)])
                    lines.append([((x0 + x1)/2, base_y + dist), (x1, base_y)])
                else:  # truncated top
                    dy = y1 - base_y
                    if dy > 0:
                        lines.append([(x0, base_y), (x0 + (x1-x0)/2 * dy/dist, y1)])
                        lines.append([(x1 - (x1-x0)/2 * dy/dist, y1), (x1, base_y)])
            elif direction == "down":
                base_y = y1 - i * dist
                if base_y - dist >= y0:  # full triangle
                    lines.append([(x0, base_y), ((x0 + x1)/2, base_y - dist)])
                    lines.append([((x0 + x1)/2, base_y - dist), (x1, base_y)])
                else:  # truncated bottom
                    dy = base_y - y0
                    if dy > 0:
                        lines.append([(x0, base_y), (x0 + (x1-x0)/2 * dy/dist, y0)])
                        lines.append([(x1 - (x1-x0)/2 * dy/dist, y0), (x1, base_y)])

    elif direction in ("left", "right"):
        x = x1 - x0
        for i in range(int(x // dist) + 1):
            if direction == "right":
                base_x = i * dist + x0
                if base_x + dist <= x1:
                    lines.append([(base_x, y0), (base_x + dist, (y0+y1)/2)])
                    lines.append([(base_x + dist, (y0+y1)/2), (base_x, y1)])
                else:
                    dx = x1 - base_x
                    if dx > 0:
                        lines.append([(base_x, y0), (x1, y0 + (y1-y0)/2 * dx/dist)])
                        lines.append([(x1, y1 - (y1-y0)/2 * dx/dist), (base_x, y1)])
            elif direction == "left":
                base_x = x1 - i * dist
                if base_x - dist >= x0:
                    lines.append([(base_x, y0), (base_x - dist, (y0+y1)/2)])
                    lines.append([(base_x - dist, (y0+y1)/2), (base_x, y1)])
                else:
                    dx = base_x - x0
                    if dx > 0:
                        lines.append([(base_x, y0), (x0, y0 + (y1-y0)/2 * dx/dist)])
                        lines.append([(x0, y1 - (y1-y0)/2 * dx/dist), (base_x, y1)])

    # Add all lines as a single collection for speed
    lc = LineCollection(lines, colors=color, linewidths=linewidth, alpha=alpha, zorder=zorder)
    ax.add_collection(lc)
    return lc

class HandlerHatch(HandlerTuple):
    def __init__(self, **kwargs):
        super().__init__(ndivide=None, **kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        """
        Draw both Rectangle and LineCollection overlaid in the legend box.
        """
        patch, hatch = orig_handle

        # Scale patch to legend box
        p = Rectangle([xdescent, ydescent], width, height,
                      facecolor=patch.get_facecolor(),
                      edgecolor=patch.get_edgecolor(),
                      transform=trans)

        # Transform hatch line segments into legend box
        lines = []
        for seg in hatch.get_segments():
            seg2 = [(xdescent + x * width, ydescent + y * height) for (x, y) in seg]
            lines.append(seg2)

        h = LineCollection(lines,
                           colors=hatch.get_colors(),
                           linewidths=hatch.get_linewidths(),
                           transform=trans)

        return [h, p]
    
def hatched_legend_entry(facecolor, direction="up", hatch_color="k", lw=globals.hatch_linewidth, zorder=-1):
    """
    Create a proxy legend handle: a colored rectangle with triangular hatching.
    The hatching is drawn inside the normalized legend box [0,1]x[0,1].
    
    direction: "up", "down", "left", "right"
    """
    # Background rectangle
    patch = Rectangle((0, 0), 1, 1, facecolor=facecolor, edgecolor="black")

    # Hatch lines in legend coordinates
    if direction == "up":
        lines = [[(0, 0), (0.5, 1)], [(0.5, 1), (1, 0)]]
    elif direction == "down":
        lines = [[(0, 1), (0.5, 0)], [(0.5, 0), (1, 1)]]
    elif direction == "left":
        lines = [[(1, 0), (0, 0.5)], [(0, 0.5), (1, 1)]]
    elif direction == "right":
        lines = [[(0, 0), (1, 0.5)], [(1, 0.5), (0, 1)]]
    else:
        lines = []

    hatch = LineCollection(lines, colors=hatch_color, linewidths=lw, zorder=zorder)

    return (patch, hatch)

def boxplot(
    df,
    ci=None,
    label=None,
    figsize=None,
    dpi=100,
    axis=None,
    new_coloring=globals.boxplot_new_coloring,
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
    new_coloring : bool, optional
        determines if hashed dataset combinations are used to derive a 
        colorscheme, or classic blue, white, red is used
        'old' -> blue, white, red
        'new' -> hashed dataset combinations + hatching

    Returns
    -------
    fig : matplotlib.figure.Figure
        the boxplot
    ax : matplotlib.axes.Axes
    """
    values = df.copy()
    if "dataset" in values.columns:
        unique_combos = values["dataset"].unique()
    elif "validation" in values.columns:
        values["dataset"] = values["validation"]
        unique_combos = values["dataset"].unique()
    else:
        raise ValueError("Boxplot has yet to be implemented for this case")
    # make plot

    axes = [axis]
    if axis is None:
        n_axes = ((len(unique_combos)-1)//globals.n_boxplots_in_row) + 1
        if values["dataset"].nunique() > globals.bin_th: 
            dims = [
                globals.boxplot_width_vertical*values["dataset"].nunique()/globals.bin_th \
                    if values["dataset"].nunique()<globals.n_boxplots_in_row \
                        else globals.boxplot_width_vertical*globals.n_boxplots_in_row/globals.bin_th,
                globals.boxplot_height_vertical*(n_axes**(1/1.4))
            ]
        else:
            dims = [
                globals.boxplot_width_vertical,
                globals.boxplot_height_vertical*(n_axes**(1/1.4))
            ]

        fig = plt.figure(figsize = (dims[0], dims[1]))
        axes = []
        for i in range(n_axes):
            ax_left = globals.ax_left
            ax_bottom = (n_axes - i - 1 + globals.ax_bottom)/n_axes
            ax = fig.add_axes([ax_left, ax_bottom, globals.ax_width, globals.ax_height/n_axes])
            axes.append(ax)
    else:
        fig = None
    # styling of the boxes
    kwargs = {"patch_artist": True}
    for key, value in plotting_kwargs.items():
        kwargs[key] = value

    if not 'widths' in kwargs:
        widths = 0.8
    else:
        widths = kwargs['widths']
        del kwargs['widths']

    # changes necessary to have confidence intervals in the plot
    # could be an empty list or could be 'None', if de-selected from the kwargs
    if ci:
        widths = widths/2
        widths_ci = widths/2
        spacing = (widths + widths_ci)/2

    palette = get_palette_for(unique_combos)
    l_low, l_up = [], []

    for ax_i in range(n_axes): #determine which ax to draw on
        ax = axes[ax_i]
        ax_combos = unique_combos[ax_i*globals.n_boxplots_in_row:(ax_i+1)*globals.n_boxplots_in_row]
        for i, d in enumerate(ax_combos): #slice combos to obtain only combinations on axis
            data = values[values["dataset"]==d]
            position = i
            if ci:
                pos_lower = position - spacing
                pos_upper = position + spacing
            color_cen = palette[ax_combos[i]] if new_coloring else "white"
            n_lines = len(ax.lines)
            cen = sns.boxplot(data = data, 
                            x = "label",
                            y = "value",
                            positions = [position],
                            color = color_cen,
                            showfliers=False,
                            widths=widths,
                            ax=ax, 
                            orient="v",
                            dodge=True,
                            **kwargs)
            capsizing(cen, n_lines = n_lines)

            if ci:
                c_lower = palette[ax_combos[i]] if new_coloring else"#87CFEBAA"
                c_upper = palette[ax_combos[i]] if new_coloring else'#FF6347AA'
                n_lines = len(ax.lines)
                low = sns.boxplot(data = ci[d],
                                y = "lower",
                                positions = [pos_lower],
                                color = c_lower,
                                showfliers=False,
                                widths=widths_ci,
                                ax=ax, 
                                orient="v",
                                dodge=True,
                                **kwargs)
                capsizing(low, n_lines=n_lines)
                l_low.append(ax.patches[-1]) # ax.patches[-1] gets last drawn patch

                n_lines = len(ax.lines)
                up = sns.boxplot(data = ci[d],
                                y = "upper",
                                positions = [pos_upper],
                                color = c_upper,
                                showfliers=False,
                                widths=widths_ci,
                                ax=ax, 
                                orient="v",
                                dodge=True,
                                **kwargs)     
                capsizing(up, n_lines=n_lines)
                l_up.append(ax.patches[-1])               

        if label is not None:
            plt.ylabel(label, fontsize = globals.fontsize_label)
            #insert xlabel here

        if ci and new_coloring:
            dist = (ax.get_ylim()[1]-ax.get_ylim()[0])/globals.num_hatches
            for low in l_low:
                triangle_hatching(ax, low, dist=dist, direction="down", color=low.get_facecolor()[:3], linewidth=globals.hatch_linewidth) 
                low.set_facecolor(low.get_facecolor()[:3]+(globals.ci_alpha,))
            for up in l_up:
                triangle_hatching(ax, up, dist=dist, direction="up", color=up.get_facecolor()[:3], linewidth=globals.hatch_linewidth)
                up.set_facecolor(up.get_facecolor()[:3]+(globals.ci_alpha,))

            up_handle   = hatched_legend_entry(up.get_facecolor()[:3]+(globals.ci_alpha,), hatch_color=up.get_facecolor()[:3], direction="up", lw=globals.hatch_linewidth)
            low_handle  = hatched_legend_entry(low.get_facecolor()[:3]+(globals.ci_alpha,), hatch_color=low.get_facecolor()[:3], direction="down", lw=globals.hatch_linewidth)

            ax.legend(handles=[up_handle, low_handle],
                    labels=["Upper CI", "Lower CI"],
                    fontsize=globals.fontsize_legend,
                    loc=best_legend_pos_exclude_list(ax),
                    handler_map={tuple: HandlerHatch()})
        ax.set_xlabel(None)

        if ci and not new_coloring:
            low_patch = Patch(facecolor=c_lower, edgecolor="black")
            up_patch = Patch(facecolor=c_upper, edgecolor="black")

            ax.legend(handles=[up_patch, low_patch],
                    labels=["Upper CI", "Lower CI"],
                    fontsize=globals.fontsize_legend,
                    loc=best_legend_pos_exclude_list(ax))

        positions = np.arange(len(ax_combos))
        ticklabels = values["label"].unique()[ax_i*globals.n_boxplots_in_row:(ax_i+1)*globals.n_boxplots_in_row]
        ax.set_xticks(positions)
        ax.set_xticklabels(ticklabels)
        ax.tick_params(labelsize=globals.fontsize_ticklabel)

        ticks = ax.get_xticks()
        midpoints = [(ticks[i] + ticks[i + 1]) / 2 for i in range(len(ticks) - 1)]
        ax.xaxis.set_minor_locator(plt.FixedLocator(midpoints))
        ax.grid(which='minor', color='gray', linestyle='dotted', linewidth=0.8)
        ax.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.8)
        ax.grid(which='major', axis='x', visible=False)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if len(ticks)>globals.no_growth_th_v:
            ax.set_xlim(ticks[0]-(ticks[1]-ticks[0])/2, ticks[-1]+(ticks[-1]-ticks[-2])/2)
        else:
            ax.set_xlim((ticks[0]+ticks[-1])/2-(globals.no_growth_th_v+1)/2, (ticks[0]+ticks[-1])/2 +(globals.no_growth_th_v+1)/2)

        if not ci:
            ax.legend([],[], fontsize=globals.fontsize_legend, loc=best_legend_pos_exclude_list(ax))

    return fig, axes

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
    # Same Edgecolor and Edgewidth as boxplots
    values[values.keys()[0]].value_counts().sort_index().plot.bar(ax=ax,
                                                                  color=color,
                                                                  edgecolor=globals.boxplot_edgecolor,
                                                                  linewidth=globals.boxplot_edgewidth)

    ax.tick_params(labelsize=globals.fontsize_ticklabel)
    ax.grid(axis='y')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels(tick_labels, rotation=45)

    plt.ylabel(label, fontsize=globals.fontsize_label)

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
    while bin_size < min_size and nbins > 1:
        nbins -= 1
        bin_values, unique_values, bin_size = resize_bins(sorted, nbins)

    # use metadata to sort dataframe
    df = pd.concat([df, metadata_values], axis=1).sort_values(meta_key)
    df.drop(columns=meta_key, inplace=True)

    # put binned data in dictionary
    binned = {}
    for bin in unique_values:
        bin_index = np.where(bin_values == bin)
        bin_sorted = sorted[bin_index]
        bin_df = df.iloc[bin_index]

        bin_label = "{:.2f}-{:.2f} {}".format(
            float(np.min(bin_sorted)), float(np.max(bin_sorted)), meta_units
        )

        # check column counts (at least min_size values in each)
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


def aggregate_subplots(to_plot: dict, funct, **kwargs):
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
    **kwargs: dict
        arguments to pass on to the plotting function

    Return
    ------
    fig, axes
    """
    sub_n = len(to_plot.keys())
    if sub_n > globals.max_subplots:
        warnings.warn(
            f"Number of subplots ({sub_n}) exceeds maximum allowed ({globals.max_subplots}). "
            "Plot creation skipped.",
            UserWarning
        )
        return None, None
    if sub_n == 1:
        for n, (bin_label, data) in enumerate(to_plot.items()):
            # fig = plt.figure()
            # ax = fig.add_axes([globals.ax_left, globals.ax_bottom, globals.ax_width, globals.ax_height])
            fig, ax = funct(df=data,**kwargs)
            fig.set_figheight(globals.boxplot_height_horizontal)
            fig.set_figwidth(globals.boxplot_width_horizontal)
            ax.set_ylabel("") # Labels get added later as superlabels
            ax.set_xlabel("")
    elif sub_n > 1:
        n_col = globals.n_col_agg
        n_rows = int(np.ceil(sub_n / n_col))
        fig = plt.figure()
        for n, (bin_label, data) in enumerate(to_plot.items()):
            ax_left = (n % n_col + globals.ax_left)/n_col
            ax_bottom = (n_rows - (n//n_col) - 1 + globals.ax_bottom)/n_rows
            ax = fig.add_axes([ax_left, ax_bottom, globals.ax_width/n_col, globals.ax_height/n_rows])
            # Make sure funct has the correct parameters format
            if 'axis' not in funct.__code__.co_varnames:
                raise KeyError(
                    "'axis' should be in the parameters of the given function {}"
                    .format(funct))
            funct(df=data, axis=ax, **kwargs)
            ax.set_title(bin_label, fontdict={"fontsize": globals.fontsize_label}) # because they are subheadings
            if n != 0: # empties extra legends
                ax.legend([], [], frameon=False, fontsize=globals.fontsize_legend)
        fig.set_figheight(globals.boxplot_height_vertical*n_rows)
        fig.set_figwidth(globals.boxplot_width_vertical*n_col)
    
    return fig, np.array(fig.axes)


def bplot_multiple(to_plot, **kwargs) -> tuple:
    """
    Create subplots for each metadata category/range

    Parameters
    ----------
    to_plot : dict
        dictionary of {'bin name': Dataframe}
    """

    if "axis" in kwargs.keys():
        del kwargs["axis"]

    fig, axes = aggregate_subplots(to_plot=to_plot,
                                   funct=boxplot,
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
                  axis_name,
                  metadata_name,
                  axis=None,
                  **kwargs) -> tuple:
    """
    Create individual plot with grouped boxplots by metadata value

    Parameters
    ----------
    to_plot: pd.Dataframe
        Seaborn-formatted dataframe
    axis_name: str
        Name of the value-axis
    metadata_name: str
        Name of the metadata type
    axis : matplotlib.axes.Axis, optional
        if provided, the function will create the plot on the specified axis
    """
    labels = None
    return_figax = False
    orient = "v"
    n_meta = to_plot[metadata_name].nunique()
    if axis is None:
        return_figax = True
        if len(set(to_plot[metadata_name])) > globals.orient_th:
            orient = "h"
        if orient == "h":
            if n_meta > globals.meta_bin_th: 
                dims = [
                    globals.boxplot_width_horizontal,
                    globals.boxplot_height_horizontal*n_meta/globals.meta_bin_th
                ]
            else:
                dims = [
                    globals.boxplot_width_horizontal,
                    globals.boxplot_height_horizontal
                ]

        if orient == "v":
            dims = [globals.boxplot_width_vertical,
                    globals.boxplot_height_vertical
            ]

        fig = plt.figure(figsize = (dims[0], dims[1]))
        axis = fig.add_axes([globals.ax_left, globals.ax_bottom, globals.ax_width, globals.ax_height])
        
    if orient == "v":
        x = metadata_name
        y = "values"
    elif orient == "h":
        x = "values"
        y = metadata_name

        # add N points to the axis labels
    to_plot = add_cat_info(to_plot, metadata_name=metadata_name).sort_values("Dataset", ascending=True)
    
    if not 'widths' in kwargs:
        # Automatically size boxplot width to number of Datasetcombinations
        widths = 0.8/to_plot.Dataset.nunique()

    unique_combos = to_plot.set_index(np.arange(to_plot.index.size))["Dataset"].unique()
    palette = get_palette_for(unique_combos)

    box = sns.boxplot(
        x=x,
        y=y,
        hue="Dataset",
        data=to_plot.set_index(np.arange(to_plot.index.size)),
        palette=palette,
        ax=axis,
        showfliers=False,
        orient=orient,
        widths=widths,
        dodge=True
    )
    capsizing(box, orient=orient, iterative=False)

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
            palette=palette,  # Same palette as boxplot
            ax=axis,
            size=point_size,         # Point size
            dodge=True,     # This aligns the points with their respective boxes
            jitter=False,   # Disable jitter to keep points centered
            orient=orient,
            legend=False    # Avoid duplicate legend
        )

    # needed for overlapping station names
    box.tick_params(labelsize=globals.fontsize_ticklabel)

    # change y-labels to one line, so they don't get crwoded
    if n_meta >= 10:
        y_labels = [label.get_text() for label in box.get_yticklabels()]
        y_labels_fixed = [label.replace("\n", ", ") for label in y_labels]
        box.set_yticklabels(y_labels_fixed, fontsize=globals.fontsize_ticklabel)

    if orient == "v":
        axis.set_ylabel(axis_name, fontsize=globals.fontsize_label)
        axis.xaxis.label.set_fontsize(globals.fontsize_label)

        ticks = axis.get_xticks()
        midpoints = [(ticks[i] + ticks[i + 1]) / 2 for i in range(len(ticks) - 1)]
        axis.xaxis.set_minor_locator(plt.FixedLocator(midpoints))
        axis.grid(which='minor', color='gray', linestyle='dotted', linewidth=0.8)
        axis.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.8)
        axis.grid(which='major', axis='x', visible=False)

        if len(ticks)>globals.no_growth_th_v:
            axis.set_xlim(ticks[0]-(ticks[1]-ticks[0])/2, ticks[-1]+(ticks[-1]-ticks[-2])/2)
        else:
            axis.set_xlim((ticks[0]+ticks[-1]-globals.no_growth_th_v-1)/2, (ticks[0]+ticks[-1]+globals.no_growth_th_v+1)/2)

    if orient == "h":
        axis.set_xlabel(axis_name, fontsize=globals.fontsize_label)
        axis.yaxis.label.set_fontsize(globals.fontsize_label)

        ticks = axis.get_yticks()
        midpoints = [(ticks[i] + ticks[i + 1]) / 2 for i in range(len(ticks) - 1)]
        axis.yaxis.set_minor_locator(plt.FixedLocator(midpoints))
        axis.grid(which='minor', color='gray', linestyle='dotted', linewidth=0.8)
        axis.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.8)
        axis.grid(which='major', axis='y', visible=False)

        if len(ticks)>globals.no_growth_th_h:
            axis.set_ylim(ticks[0]-(ticks[1]-ticks[0])/2, ticks[-1]+(ticks[-1]-ticks[-2])/2)
        else:
            axis.set_ylim((ticks[0]+ticks[-1]-globals.no_growth_th_h-1)/2, (ticks[0]+ticks[-1]+globals.no_growth_th_h+1)/2)

    axis.set_axisbelow(True)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.legend(loc=best_legend_pos_exclude_list(axis), fontsize=globals.fontsize_legend)

    if return_figax:
        #fig.set_figwidth(dims[0])
        #fig.set_figheight(dims[1])

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
        offset of logo
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
        axis_name=ax_label,
        metadata_name=meta_key,
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
    is_scattered: Optional[bool] = False,
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
                cmap.set_under("pink")
            if mask_over is not None:
                v_max = mask_over
                cmap.set_over("pink")

    # initialize plot
    fig, ax = init_plot(figsize, dpi, projection)

    # scatter point or mapplot
    if ref_short in globals.scattered_datasets or is_scattered:  # scatter
        if not plot_extent:
            plot_extent = get_plot_extent(df)
        df = df.groupby(["lat", "lon"]).mean() # Because One Station can have multiple sensors the average value between these sensors is taken
        df_na = df[df.isna()]
        df_num = df[df.notna()]
        lat, lon, gpi = globals.index_names
        x, y = df_num.index.get_level_values(lon), df_num.index.get_level_values(lat)
        im = ax.scatter(x,
                        y,
                        c=df_num,
                        cmap=cmap,
                        s=globals.min_markersize,
                        vmin=v_min,
                        vmax=v_max,
                        edgecolors='black',
                        linewidths=0.7,
                        zorder=5,
                        transform=globals.data_crs,
                        label="Values computed")
        im_nan = ax.scatter(df_na.index.get_level_values(lon), 
                            df_na.index.get_level_values(lat),
                            c="k",
                            marker=".",
                            s=globals.nan_markersize,
                            zorder=4,
                            transform=globals.data_crs,
                            label="No value computed") # represent nan values as a dot to not confuse them with 0-values
        if metric == 'status':
            ax.legend(handles=[
                Patch(facecolor=cls[x], label=labs[x])
                for x in range(len(globals.status)) if (x - 1) in vals
            ],
                      loc='lower center',
                      ncol=4,
                      fontsize=globals.fontsize_legend)

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
                      ncol=4,
                      fontsize=globals.fontsize_legend)

    style_map(ax, plot_extent, **style_kwargs)
    if ref_short in globals.scattered_datasets:
        if len(df) < 400: # For a high amount of points the minimum markersize is kept
            s = non_overlapping_markersize(ax, im)
            im.set_sizes([s])   

    if add_cbar:  # colorbar
        fig, im, cax = _make_cbar(fig,
                             ax,
                             im,
                             ref_short,
                             metric,
                             label=label,
                             diff_map=diff_map,
                             scl_short=scl_short)

    #if legend wasn't created yet creat one  
    if (ax.get_legend() is None) and (ref_short in globals.scattered_datasets):
        ax.legend(borderpad = 0.6)
        
    return fig, ax

def plot_spatial_extent(
    polys: dict,
    ref_points: bool = None,
    overlapping: bool = False,
    intersection_extent: tuple = None,
    reg_grid=False,
    grid_stepsize=None,
    is_scattered=False,
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
    fig, ax = init_plot(figsize=globals.map_figsize, dpi=globals.dpi_min)
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
        if reg_grid and not is_scattered:
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
    title_style = {"fontsize": globals.fontsize_title}
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
               fontsize=globals.fontsize_legend,
               framealpha=0.95,
               facecolor="white",
               edgecolor="white")
    plt.tight_layout()


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
                                           logo_size: float) -> Tuple:
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
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The scaled figure object.
    ax : matplotlib.axes.Axes
        The axes object, potentially modified.
    scale : float
        The modified scale value.
    """
    factor = 1

    suptitle_text = fig._suptitle
    suptitle_text.set_fontsize(min(globals.fontsize_title_max, suptitle_text.get_fontsize() * factor))

    for tick in ax.get_xticklabels():
        tick.set_fontsize(min(globals.fontsize_ticklabel_max, tick.get_fontsize() * factor))

    for tick in ax.get_yticklabels():
        tick.set_fontsize(min(globals.fontsize_ticklabel_max, tick.get_fontsize() * factor))

    xlabel = ax.xaxis.label
    ylabel = ax.yaxis.label
    xlabel.set_fontsize(min(globals.fontsize_label_max, xlabel.get_fontsize() * factor))
    ylabel.set_fontsize(min(globals.fontsize_label_max, ylabel.get_fontsize() * factor))

    legend = ax.get_legend()

    legend.get_title().set_fontsize(min(globals.fontsize_legend_max, legend.get_title().get_fontsize() * factor))

    for text in legend.get_texts():
        text.set_fontsize(min(globals.fontsize_legend_max, text.get_fontsize() * factor))

    for patch in legend.get_patches():
        patch_size = 1.0 * (factor)  # Scale factor based on n_networks
        patch.set_height(patch.get_height() * patch_size)

    return fig, ax

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
        self.label_template = "{dataset_name} [{unit}]"

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
            _fig = plt.figure(figsize=(globals.boxplot_height_vertical, globals. boxplot_width_vertical))

        if not incl_median_iqr_n_axs:
            ax_box = _fig.add_axes([globals.ax_left, globals.ax_bottom, globals.ax_width, globals.ax_height])
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
                _ax.tick_params(labelsize=globals.fontsize_ticklabel)
                _ax.spines['right'].set_visible(False)
                _ax.spines['top'].set_visible(False)
            except AttributeError:
                pass

        return ClusteredBoxPlotContainer(fig=_fig,
                                         ax_box=ax_box,
                                         ax_median=ax_median,
                                         ax_iqr=ax_iqr,
                                         ax_n=ax_n)
