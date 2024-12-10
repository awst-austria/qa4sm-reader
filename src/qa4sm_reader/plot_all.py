# %%
# # -*- coding: utf-8 -*-
import os
from typing import Union, List, Tuple, Dict
from itertools import chain

import pandas as pd
from qa4sm_reader.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber
from qa4sm_reader.plotter import QA4SMPlotter, QA4SMCompPlotter
from qa4sm_reader.img import QA4SMImg
import qa4sm_reader.globals as globals
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_all(filepath: str,
             temporal_sub_windows: List[str] = None,
             metrics: list = None,
             extent: tuple = None,
             out_dir: str = None,
             out_type: str = 'png',
             save_all: bool = True,
             save_metadata: Union[str, bool] = 'never',
             save_csv: bool = True,
             engine: str = 'h5netcdf',
             **plotting_kwargs) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Creates boxplots for all metrics and map plots for all variables.
    Saves the output in a folder-structure.

    Parameters
    ----------
    filepath : str
        path to the *.nc file to be processed.
    temporal_sub_windows : List[str], optional (default: None)
        List of temporal sub-windows to be processed. If None, all periods present are automatically extracted from the file.
    metrics : set or list, optional (default: None)
        metrics to be plotted. If None, all metrics with data are plotted
    extent : tuple, optional (default: None)
        Area to subset the values for -> (min_lon, max_lon, min_lat, max_lat)
    out_dir : str, optional (default: None)
        Path to output generated plot. If None, defaults to the current working directory.
    out_type: str or list
        extensions which the files should be saved in
    save_all: bool, optional (default: True)
        all plotted images are saved to the output directory
    save_metadata: str or bool, optional (default: 'never')
        for each metric, metadata plots are provided
        (see plotter.QA4SMPlotter.plot_save_metadata)
        - 'never' or False: No metadata plots are created
        - 'always': Metadata plots are always created for all metrics
                   (set the meta_boxplot_min_size to 0)
        - 'threshold' or True: Metadata plots are only created if the number
                               of points is above the `meta_boxplot_min_size`
                               threshold from globals.py. Otherwise a warning
                               is printed.
    save_csv: bool, optional. Default is True.
        save a .csv file with the validation statistics
    engine: str, optional (default: h5netcdf)
        Engine used by xarray to read data from file. For qa4sm this should
        be h5netcdf.
    plotting_kwargs: arguments for plotting functions.

    Returns
    -------
    fnames_boxplots: list
    fnames_mapplots: list
        lists of filenames for created mapplots and boxplots
    fnames_csv: list
    fnames_cbplot: list
        list of filenames for created comparison boxplots
    """


    if isinstance(save_metadata, bool):
        if not save_metadata:
            save_metadata = 'never'
        else:
            save_metadata = 'threshold'
    save_metadata = save_metadata.lower()

    _options = ['never', 'always', 'threshold']
    if save_metadata not in _options:
        raise ValueError(f"save_metadata must be one of: "
                         f"{', '.join(_options)} "
                         f"but `{save_metadata}` was passed.")

    # initialise image and plotter
    fnames_bplot, fnames_mapplot, fnames_csv = [], [], []

    comparison_periods = None
    if temporal_sub_windows is None:
        periods = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(filepath)
    else:
        periods = np.array(temporal_sub_windows)
    # Filter out all items that are purely digits 
    # Needs to be here because when qa4sm-validaion is run the temporal_sub_windows is not None
    comparison_periods = periods
    periods = [period for period in periods if not period.isdigit()]

    for period in periods:
        print(f'period: {period}')
        img = QA4SMImg(
            filepath,
            period=period,
            extent=extent,
            ignore_empty=True,
            engine=engine,
        )
        plotter = QA4SMPlotter(
            image=img,
            out_dir=os.path.join(out_dir, str(period)) if period else out_dir)

        if metrics is None:
            metrics = img.metrics

        # iterate metrics and create files in output directory
        for metric in metrics:
            metric_bplots, metric_mapplots = plotter.plot_metric(
                metric=metric,
                period=period,
                out_types=out_type,
                save_all=save_all,
                **plotting_kwargs)
            # there can be boxplots with no mapplots
            if metric_bplots:
                fnames_bplot.extend(metric_bplots)
            if metric_mapplots:
                fnames_mapplot.extend(metric_mapplots)
            if img.metadata and (save_metadata != 'never'):
                if save_metadata == 'always':
                    kwargs = {'meta_boxplot_min_samples': 0}
                else:
                    kwargs = {
                        'meta_boxplot_min_samples':
                        globals.meta_boxplot_min_samples
                    }

                if period == globals.DEFAULT_TSW:
                    kwargs['period'] = globals.DEFAULT_TSW

                fnames_bplot.extend(
                    plotter.plot_save_metadata(metric,
                                               out_types=out_type,
                                               **kwargs))

        if save_csv:
            out_csv = plotter.save_stats(period=period)
            fnames_csv.append(out_csv)

    #$$
    # ? move somewhere else?
    fnames_cbplot = []
    if isinstance(out_type, str):
        out_type = [out_type]
    metrics_not_to_plot = list(set(chain(globals._metadata_exclude, globals.metric_groups['triple'], ['n_obs']))) # metadata, tcol metrics, n_obs
    if globals.DEFAULT_TSW in comparison_periods and len(comparison_periods) > 1:
        #check if stability metrics where calculated
        stability = all(item.isdigit() for item in comparison_periods if item != 'bulk')
        cbp = QA4SMCompPlotter(filepath)
        comparison_boxplot_dir = os.path.join(out_dir, globals.CLUSTERED_BOX_PLOT_OUTDIR)
        os.makedirs(comparison_boxplot_dir, exist_ok=True)

        for available_metric in cbp.metric_kinds_available:
            if available_metric in metrics.keys(
            ) and available_metric not in metrics_not_to_plot:

                spth = [Path(out_dir) / globals.CLUSTERED_BOX_PLOT_OUTDIR /
                        f'{globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=available_metric, filetype=_out_type)}'
                        for _out_type in out_type]
                _fig = cbp.plot_cbp(
                    chosen_metric=available_metric,
                    out_name=spth,
                    stability=stability
                )
                plt.close(_fig)
                fnames_cbplot.extend(spth)

    return fnames_bplot, fnames_mapplot, fnames_csv, fnames_cbplot


def get_img_stats(
    filepath: str,
    extent: tuple = None,
) -> pd.DataFrame:
    """
    Creates a dataframe containing summary statistics for each metric

    Parameters
    ----------
    filepath : str
        path to the *.nc file to be processed.
    extent : list
        list(x_min, x_max, y_min, y_max) to create a subset of the values

    Returns
    -------
    table : pd.DataFrame
        Quick inspection table of the results.
    """
    img = QA4SMImg(filepath, extent=extent, ignore_empty=True)
    table = img.stats_df()

    return table
