from typing import Optional, Tuple, Dict, Literal
from qa4sm_reader.plotting_methods import (_replace_status_values, init_plot,
                                           get_plot_extent, Patch,
                                           geotraj_to_geo2d, _make_cbar,
                                           style_map)
import copy
from qa4sm_reader import globals
import pandas as pd
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import xarray as xr
import seaborn as sns
import re
import numpy as np

sns.set_context("notebook")


status = {
    -1: 'Other error',
    0: 'Success',
    1: 'Not enough data',
    2: 'Metric calculation failed',
    3: 'Temporal matching failed',
    4: 'No overlap for temporal match',
    5: 'Scaling failed',
    6: 'Unexpected validation error',
    7: 'Missing GPI data',
    8: 'Data reading failed'
}

metric_pretty_names = {
    'R': 'Pearson\'s r',
    'R_ci_lower': 'Pearson\'s r lower confidence interval',
    'R_ci_upper': 'Pearson\'s r upper confidence interval',
    'p_R': 'Pearson\'s r p-value',
    'RMSD': 'Root-mean-square deviation',
    'BIAS': 'Bias (difference of means)',
    'BIAS_ci_lower': 'Bias (difference of means) lower confidence interval',
    'BIAS_ci_upper': 'Bias (difference of means) upper confidence interval',
    'n_obs': '# observations',
    'urmsd': 'Unbiased root-mean-square deviation',
    'urmsd_ci_lower': 'Unbiased root-mean-square deviation lower confidence interval',
    'urmsd_ci_upper': 'Unbiased root-mean-square deviation upper confidence interval',
    'RSS': 'Residual sum of squares',
    'mse': 'Mean square error',
    'mse_corr': 'Mean square error correlation',
    'mse_bias': 'Mean square error bias',
    'mse_var': 'Mean square error variance',
    'snr': 'Signal-to-noise ratio',
    'snr_ci_lower': 'Signal-to-noise ratio lower confidence interval',
    'snr_ci_upper': 'Signal-to-noise ratio upper confidence interval',
    'err_std': 'Error standard deviation',
    'err_std_ci_lower': 'Error standard deviation lower confidence interval',
    'err_std_ci_upper': 'Error standard deviation upper confidence interval',
    'beta': 'TC scaling coefficient',
    'beta_ci_lower': 'TC scaling coefficient lower confidence interval',
    'beta_ci_upper': 'TC scaling coefficient upper confidence interval',
    'rho': 'Spearman\'s ρ',
    'rho_ci_lower': 'Spearman\'s ρ lower confidence interval',
    'rho_ci_uppper': 'Spearman\'s ρ upper confidence interval',
    'p_rho': 'Spearman\'s ρ p-value',
    'tau': 'Kendall rank correlation',
    'p_tau': 'Kendall tau p-value',
    'status': 'Validation success status',
    # 'tau': 'Kendall rank correlation',        # currently QA4SM is hardcoded not to calculate kendall tau
    # 'p_tau': 'Kendall tau p-value',
    'slopeR' : 'Theil-Sen slope of R',
    'slopeURMSD' : 'Theil-Sen slope of urmsd',
    'slopeBIAS' : 'Theil-Sen slope of BIAS'
}
metric_value_ranges = {  # from /qa4sm/validator/validation/graphics.py
    'R': [-1, 1],
    'R_ci_lower': [-1, 1],
    'R_ci_upper': [-1, 1],
    'p_R': [0, 1],
    # probability that observed correlation is statistical fluctuation
    'rho': [-1, 1],
    'rho_ci_lower': [-1, 1],
    'rho_ci_upper': [-1, 1],
    'p_rho': [0, 1],
    'tau': [-1, 1],
    'p_tau': [0, 1],
    'RMSD': [0, None],
    'BIAS': [None, None],
    'BIAS_ci_lower': [None, None],
    'BIAS_ci_upper': [None, None],
    'n_obs': [0, None],
    'urmsd': [0, None],
    'urmsd_ci_lower': [0, None],
    'urmsd_ci_upper': [0, None],
    'RSS': [0, None],
    'mse': [0, None],
    'mse_corr': [0, None],
    'mse_bias': [0, None],
    'mse_var': [0, None],
    'snr': [None, None],
    'snr_ci_lower': [None, None],
    'snr_ci_upper': [None, None],
    'err_std': [None, None],
    'err_std_ci_lower': [None, None],
    'err_std_ci_upper': [None, None],
    'beta': [None, None],
    'beta_ci_lower': [None, None],
    'beta_ci_upper': [None, None],
    'status': [-1, len(status) - 2],
    'slopeR': [None, None],
    'slopeURMSD': [None, None],
    'slopeBIAS': [None, None],
}


def select_column_by_all_keywords(dataframe: pd.DataFrame, datasets: list,
                                  metric: str, datasets_in_df: list) -> str:
    """
    Select a column name from a dataframe based on the presence of all
    keywords.

    Parameters:
    dataframe (pd.DataFrame): The dataframe from which to select the column.
    keywords (list): A list of keywords that must all be present in the
    column name.

    Returns:
    str: The name of the first column that matches all the keywords. None if
    no column matches.
    """
    if metric.lower().startswith("snr") or metric.lower().startswith(
            "err_std") or metric.lower().startswith("beta"):
        if len(datasets) > 1:
            raise ValueError("Only one dataset is allowed for this metric. "
                             "Please add only the dataset to the dataset "
                             "list of "
                             "which you want to get the metric value. E.g. "
                             "signal to noise ratio: ['snr'] of the dataset: "
                             "'ISMN'. The available datasets are: {}".format(
                datasets_in_df))
        else:
            pattern = r"^"+ metric.lower() +r"_\d-"+datasets[0].upper()
            for column in dataframe.columns:
                # Check if all keywords are present in the column name (case
                # insensitive)
                if re.search(pattern, column, re.IGNORECASE):
                    return column
                else:
                    pass
    else:
        for column in dataframe.columns:
            # Check if all keywords are present in the column name (case
            # insensitive)
            if all(re.search(keyword, column, re.IGNORECASE) for keyword in
                   datasets) and column.lower().startswith(metric.lower()):
                return column
            else:
                pass


class CustomPlotObject:
    """
    A class to handle NetCDF file data and plot static maps using custom_map_plot
    function.

    Attributes:
    - nc_file_path (str): Path to the NetCDF file.
    """

    def __init__(self, nc_file_path: str):
        self.nc_file_path = nc_file_path
        self.df = xr.open_dataset(nc_file_path).to_dataframe().set_index(
            ['lat', 'lon'])

    def display_metrics_and_datasets(self):
        """
        Displays available metrics and datasets for the dataset present in the
        current object.

        This method identifies valid metrics by matching keys from the
        `metric_value_ranges` dictionary to column names in the dataframe
        available in `self.df`. It also identifies valid datasets by extracting
        dataset patterns from the `nc_file_path` using a regular expression.
        The results are printed to the console.

        Attributes:
            df (pd.DataFrame): Dataframe containing data with various columns
                where metrics can potentially exist.
            nc_file_path (str): File path of the netCDF file being used, from
                which dataset patterns are extracted.
        """
        valid_metrics = [s1 for s1 in
                         list(metric_pretty_names.keys()) if any(
                s1 in s2 for s2 in list(self.df.columns))]
        dataset_pattern = r'(?<=\d-)(.*?)(?=\.)'
        valid_datasets = re.findall(dataset_pattern, self.nc_file_path)
        print(
            "The following metrics and datasets are available for this dataset:\n"
            "Datasets: ")

        for dataset in valid_datasets:
            print("- {}".format(dataset))
        print(
            """Metrics: """)
        for metric in valid_metrics:
            print("- "+metric+": "+metric_pretty_names[metric])

    def plot_map(self, metric: str, output_dir: str,
                 colormap: Optional[str] = None,
                 value_range: Optional[Tuple[float, float]] = None,
                 plotsize: Optional[Tuple[float, float]] = globals.map_figsize,
                 extent: Optional[Tuple[float, float, float, float]] = None,
                 colorbar_label: Optional[str] = None,
                 title: Optional[str] = None,
                 title_fontsize: Optional[int] = globals.fontsize_title,
                 xy_ticks_fontsize: Optional[int] = globals.fontsize_ticklabel,
                 colorbar_ticks_fontsize: Optional[int] = globals.fontsize_ticklabel,
                 dataset_list: list = None, ):
        """
        Generates a map plot for a specified metric and saves it to the
        specified
        output directory. The function uses the custom_mapplot function for
        plotting
        and performs a series of checks and validations on the input
        parameters before
        proceeding.

        Args:
            metric: The name of the metric to be plotted. Must correspond to a
                valid column in the DataFrame.
            output_dir: The directory where the output plot will be saved.
            colormap: Optional; The name of the colormap to use for the plot.
            value_range: Optional; A tuple specifying the minimum and
            maximum range
                of values for the metric to be displayed in the plot.
            plotsize: Optional; A tuple specifying the size of the output
            plot in inches.
            extent: Optional; A tuple specifying the geographical extent of
            the map.
                Must be in the format (min_longitude, max_longitude,
                min_latitude,
                max_latitude).
            colorbar_label: Optional; The label to use for the colorbar in
            the plot.
            title: Optional; The title of the map plot.
            title_fontsize: Optional; The font size of the plot title.
            colorbar_fontsize: Optional; The font size for the colorbar labels.
            dataset_list: Optional; A list of datasets to filter or match
            the metric
                data against before plotting. Used to identify the appropriate
                column in the DataFrame.

        Raises:
            ValueError: If no data is loaded in the internal DataFrame object.
            ValueError: If the specified metric is not supported or available
                in the list of predefined metric keys.
            ValueError: If no matching column is found for the specified
            metric in the
                DataFrame, or if the dataset list and metric name do not
                match a
                valid column.
        """
        if self.df is None:
            raise ValueError(
                "No data loaded. Please load a NetCDF file and convert it "
                "into a DataFrame first.")
        # Check metrics
        valid_metrics_in_dataset = [s1 for s1 in
                                    list(metric_value_ranges.keys()) if any(
                s1 in s2 for s2 in list(self.df.columns))]
        if metric not in valid_metrics_in_dataset:
            raise ValueError(
                f"Metric '{metric}' is not supported. Please choose from the "
                f"following metrics present in your dataset: "
                f"{valid_metrics_in_dataset}")
        # Check datasets
        dataset_pattern = r'(?<=\d-)(.*?)(?=\.)'
        datasets_in_df = re.findall(dataset_pattern, self.nc_file_path)
        if not all(s1 in datasets_in_df for s1 in dataset_list):
            raise ValueError(
                f"Dataset list does not match any column in the DataFrame. "
                f"Please select one of the following datasets: "
                f"{datasets_in_df}"
                f" Select at least two datasets unless you are plotting a "
                f"triple collocation metric (snr, err_std, beta).")
        column_name = select_column_by_all_keywords(self.df, dataset_list,
                                                    metric, datasets_in_df)

        if column_name is None:
            raise ValueError(
                f"Column '{metric}' does not exist in the DataFrame."
                f"Please check the dataset list and the metric name."
            )
        if globals.scattered_datasets[0] in datasets_in_df:
            ref_dataset = globals.scattered_datasets[0]
        else:
            ref_dataset = None

        custom_mapplot(
            df=self.df,
            column_name=column_name,
            ref_short=ref_dataset,
            metric=metric,
            plot_extent=extent,
            colormap=colormap,
            value_range=value_range,
            label=colorbar_label,
            title=title,
            title_fontsize=title_fontsize,
            output_dir=output_dir,
            figsize=plotsize,
            xyticks_fontsize=xy_ticks_fontsize,
            colorbar_ticks_fontsize=colorbar_ticks_fontsize,
        )


def custom_mapplot(
        df: pd.DataFrame,
        column_name: str,
        ref_short: str,
        metric: str,
        scl_short: Optional[str] = None,
        ref_grid_stepsize: Optional[float] = None,
        plot_extent: Optional[Tuple[float, float, float, float]] = None,
        colormap: Optional[str] = None,
        projection: Optional[ccrs.Projection] = None,
        add_cbar: Optional[bool] = True,
        label: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = globals.map_figsize,
        dpi: Optional[int] = globals.dpi_min,
        diff_map: Optional[bool] = False,
        value_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[str] = None,
        title: Optional[str] = None,
        title_fontsize: Optional[int] = globals.fontsize_title,
        xyticks_fontsize: Optional[int] = globals.fontsize_ticklabel,
        colorbar_ticks_fontsize: Optional[int] = globals.fontsize_ticklabel,
        **style_kwargs: Dict):
    """
        Create an overview map from df using values as color. Plots a
        scatterplot for ISMN and an image plot for other
        input values.

        Parameters
        ----------
        df : pandas.DataFrame
            input dataframe with lat, lon and values
        metric : str
            name of the metric for the plot
        ref_short : str
                short_name of the reference dataset (read from netCDF file)
        scl_short : str, default is None
            short_name of the scaling dataset (read from netCDF file).
            None if no scaling method is selected in validation.
        ref_grid_stepsize : float or None, optional (None by default)
                angular grid stepsize, needed only when ref_is_angular ==
                False,
        plot_extent : tuple or None
                (x_min, x_max, y_min, y_max) in Data coordinates. The
                default is None.
        colormap :  Colormap, optional
                colormap to be used.
                If None, defaults to globals._colormaps.
        tc_dataset_name : str, optional
                Specifies the name of the dataset for which the respective
                triple colocation metric is calculated.
        projection :  cartopy.crs, optional
                Projection to be used. If none, defaults to
                globals.map_projection.
                The default is None.
        add_cbar : bool, optional
                Add a colorbar. The default is True.
        label : str, optional
            Label of the y-axis, describing the metric. If None, a label is
            autogenerated from metadata.
            The default is None.
        figsize : tuple, optional
            Figure size in inches. The default is globals.map_figsize.
        dpi : int, optional
            Resolution for raster graphic output. The default is globals.dpi.
        diff_map : bool, default is False
            if True, a difference colormap is created
        value_range : tuple, optional
            Value range for the plot. If None, the range is determined from
            the metric_value_ranges dictionary.
        title : str, optional
            Title of the plot. The default is None.
        title_fontsize : int, optional
            Font size for the title.
        label_fontsize : int, optional
            Label font size.
        **style_kwargs :
            Keyword arguments for plotter.style_map().

        Returns
        -------
        fig : matplotlib.figure.Figure
            the boxplot
        ax : matplotlib.axes.Axes
        """
    if not colormap:
        try:
            cmap = globals._colormaps[metric]
        except:
            cmap = globals._colormaps[
                metric.split('_')[0] if '_' in metric else metric]
    else:
        cmap = colormap
    if value_range is None:
        v_min, v_max = metric_value_ranges[metric]
    else:
        v_min, v_max = value_range
    # everything changes if the plot is a difference map
    if diff_map:
        cmap = globals._diff_colormaps[
            metric.split('_')[0] if '_' in metric else metric]

    if metric == 'status':
        df = _replace_status_values(df[column_name])
        labs = list(globals.status.values())
        cls = globals.get_status_colors().colors
        vals = sorted(list(set(df.values)))
        add_cbar = False

    # No need to mask ranged in the comparison plots
    else:
        # mask values outside range (e.g. for negative STDerr from TCA)
        if metric.split('_')[0] in globals._metric_mask_range.keys():
            mask_under, mask_over = globals._metric_mask_range[
                metric.split('_')[
                    0]]  # get values from scratch to disregard quantiles
            cmap = copy.copy(cmap)
            if mask_under is not None:
                v_min = mask_under
                cmap.set_under("red")
            if mask_over is not None:
                v_max = mask_over
                cmap.set_over("red")

    # initialize plot
    fig, ax = init_plot(figsize, projection=projection)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    # scatter point or mapplot
    if ref_short in globals.scattered_datasets:  # scatter
        if not plot_extent:
            plot_extent = get_plot_extent(df)

        markersize = globals.markersize ** 2
        lat, lon, gpi = globals.index_names
        im = ax.scatter(df.index.get_level_values(lon),
                        df.index.get_level_values(lat),
                        c=df[column_name],
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
            if metric == 'status':
                plot_extent = get_plot_extent(df,
                                              grid_stepsize=ref_grid_stepsize,
                                              grid=True)
            else:
                plot_extent = get_plot_extent(df[column_name],
                                              grid_stepsize=ref_grid_stepsize,
                                              grid=True)
        if isinstance(ref_grid_stepsize, np.ndarray):
            ref_grid_stepsize = ref_grid_stepsize[0]
        if metric == 'status':
            zz, zz_extent, origin = geotraj_to_geo2d(
                df, grid_stepsize=ref_grid_stepsize)  # prep values
        else:
            zz, zz_extent, origin = geotraj_to_geo2d(
                df[column_name],
                grid_stepsize=ref_grid_stepsize)  # prep values
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
        try:
            _, _, cax = _make_cbar(fig,
                       ax,
                       im,
                       ref_short,
                       metric,
                       label=label,
                       diff_map=diff_map,
                       scl_short=scl_short,
                       wrap_text=True)
        except:
            _, _, cax = _make_cbar(fig,
                       ax,
                       im,
                       ref_short,
                       metric.split('_')[0],
                       label=label,
                       diff_map=diff_map,
                       scl_short=scl_short,
                       wrap_text=True)
        if colorbar_ticks_fontsize and cax:
            cax.tick_params(labelsize=colorbar_ticks_fontsize)
    style_map(ax, plot_extent, grid_tick_size=xyticks_fontsize, **style_kwargs)
    if title is not None and title_fontsize is not None:
        ax.set_title(title, fontsize=title_fontsize)
    elif title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{column_name}_map.png", dpi=100)
    return fig, ax
