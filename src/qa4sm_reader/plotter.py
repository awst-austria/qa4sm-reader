# -*- coding: utf-8 -*-
import re
from unittest.mock import DEFAULT
import warnings
from pathlib import Path
import os
from warnings import warn
import pandas as pd
import xarray as xr
from typing import Generator, Union, List, Tuple, Dict, Optional
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.pylab import f
import matplotlib
from matplotlib.patches import Rectangle
import seaborn as sns
from qa4sm_reader.colors import get_color_for, get_palette_for

from qa4sm_reader.img import QA4SMImg
import qa4sm_reader.globals as globals
from qa4sm_reader import plotting_methods as plm
from qa4sm_reader.plotting_methods import ClusteredBoxPlot, patch_styling, scale_figure_for_network_metadata_plot
from qa4sm_reader.exceptions import PlotterError
import qa4sm_reader.handlers as hdl
from qa4sm_reader.utils import note, filter_out_self_combination_tcmetric_vars

# Change of standard matplotlib parameters
matplotlib.rcParams['legend.framealpha'] = globals.legend_alpha
matplotlib.rcParams['boxplot.boxprops.linewidth'] = globals.boxplot_edgewidth
matplotlib.rcParams['boxplot.whiskerprops.linewidth'] = globals.boxplot_edgewidth
matplotlib.rcParams['boxplot.capprops.linewidth'] = globals.boxplot_edgewidth
matplotlib.rcParams['boxplot.medianprops.linewidth'] = globals.boxplot_edgewidth

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

class QA4SMPlotter:
    """
    Class to create image files of plots from the validation results in a QA4SMImage
    """

    def __init__(self, image: QA4SMImg, out_dir: str = None):
        """
        Create box plots from results in a qa4sm output file.

        Parameters
        ----------
        image : QA4SMImg
            The results object.
        out_dir : str, optional (default: None)
            Path to output generated plot. If None, defaults to the current working directory.
        """
        self.img = image
        self.out_dir = self.get_dir(out_dir=out_dir)

        self.ref = image.datasets.ref

        try:
            self.img.vars
        except AttributeError:
            raise PlotterError(
                "The initialized QA4SMImg object has not been loaded. 'load_data' needs "
                "to be set to 'True' in the initialization of the Image.")

    def get_dir(self, out_dir: str) -> Path:
        """Use output path if specified, otherwise same directory as the one storing the netCDF file"""
        # if out_dir and globals.DEFAULT_TSW not in out_dir:
        if out_dir:
            out_dir = Path(out_dir)  # use directory if specified
            if not out_dir.exists():
                os.makedirs(out_dir)  # make if not existing
        else:
            out_dir = self.img.filepath.parent  # use default otherwise

        return out_dir

    def _standard_filename(self, out_name: str, out_type: str = 'png') -> Path:
        """
        Standardized behaviour for filenames: if provided name has extension, it is kept; otherwise, it is saved as
        .png to self.out_dir

        Parameters
        ----------
        out_name : str
            output filename (with or without extension)
        out_type : str, optional
            contains file extensions to be plotted. If None, uses 'png'

        Returns
        -------
        outname: pathlib.Path
            correct path of the file
        """
        out_name = Path(out_name)
        # provide output directory
        out_path = self.out_dir.joinpath(out_name)

        # provide output file type
        if not out_path.suffix:
            if out_type[0] != '.':
                out_type = '.' + out_type
            out_path = out_path.with_suffix(out_type)

        return out_path

    @staticmethod
    def _box_caption(Var,
                     tc: bool = False,
                     short_caption: bool = False) -> str:
        """
        Create the dataset part of the box (axis) caption

        Parameters
        ----------
        Var: MetricVar
            variable for a metric
        tc: bool, default is False
            True if TC. Then, caption starts with "Other Data:"
        short_caption: bool, optional
            whether to use a shorter version of the caption

        Returns
        -------
        capt: str
            box caption
        """
        ref_meta, mds_meta, other_meta, _, sref_meta = Var.get_varmeta()
        ds_parts = []
        id, meta = mds_meta
        if tc:
            id, meta = other_meta  # show name of the OTHER dataset
        if short_caption:
            ds_parts.append(
                f"{ref_meta[0]} & {id}")
        else:
            ds_parts.append('Datasets: {} & {}'.format(
                ref_meta[0], id))
        capt = '\n and \n'.join(ds_parts)

        if tc:
            # Append calculation reference dataset for clearer naming
            capt = 'Other Data:\n' + capt
        return capt

    @staticmethod
    def _get_parts_name(Var, type='boxplot_basic') -> list:
        """
        Create parts for title according to the type of plot

        Parameters
        ----------
        Var: MetricVar
            variable for a metric
        type: str
            type of plot

        Returns
        -------
        parts: list of parts for title
        """
        
        parts = []
        ref, mds, other, _, sref = Var.get_varmeta()

        if type in ['boxplot_tc', 'mapplot_basic']:
            d = QA4SMPlotter._get_dataset_dict(Var) # Appends version number to pretty name if needed
            parts.extend([d[mds[0]]])
            parts.extend([d[ref[0]]])
        elif type == 'mapplot_tc':
            d = QA4SMPlotter._get_dataset_dict(Var) # Appends version number to pretty name if needed
            parts.extend([d[mds[0]]])
            for dss in Var.other_dss:
                parts.extend([d[dss[0]]])

        return parts

    @staticmethod
    def _titles_lut(type: str) -> str:
        """
        Lookup table for plot titles

        Parameters
        ----------
        type: str
            type of plot
        """
        titles = {
            'boxplot_basic':
            'Intercomparison of {}',
            'barplot_basic': 'Validation Errors',
            'mapplot_status': 'Validation Errors',
            'boxplot_tc':
            'Intercomparison of {} for {} with {}',
            'mapplot_basic':
            '{} for {} with {}',
            'mapplot_tc': '{} for {} with {} and {}',
            'metadata':
            'Intercomparison of {} by {}',
        }

        try:
            return titles[type]

        except KeyError:
            raise PlotterError(f"type '{type}' is not in the lookup table")

    @staticmethod
    def _filenames_lut(type: str) -> str:
        """
        Lookup table for file names

        Parameters
        ----------
        type: str
            type of plot
        """
        # we stick to old naming convention
        names = {
            'boxplot_basic': 'boxplot_{}',
            'barplot_basic': 'barplot_status',
            'mapplot_status': 'overview_status',
            'mapplot_common': 'overview_{}',
            'boxplot_tc': 'boxplot_{}_for_{}-{}',
            'mapplot_double': 'overview_{}-{}_and_{}-{}_{}',
            'mapplot_tc': 'overview_{}-{}_and_{}-{}_and_{}-{}_{}_for_{}-{}',
            'metadata': 'boxplot_{}_metadata_{}',
            'table': 'statistics_table',
        }

        try:
            return names[type]

        except KeyError:
            raise PlotterError(f"type '{type}' is not in the lookup table")

    def create_title(self, Var, type: str, period: str = None) -> str:
        """
        Create title of the plot

        Parameters
        ----------
        Var: MetricVar
            variable for a metric
        type: str
            type of plot
        """
        parts = [globals._metric_name[Var.metric]]
        parts.extend(self._get_parts_name(Var=Var, type=type))
        title = self._titles_lut(type=type).format(*parts)
        if period:
            if period not in globals.no_print_period:
                title = f'{period}: {title}'
        return title

    def create_filename(self, Var, type: str, period: str = None) -> str:
        """
        Create name of the file

        Parameters
        ----------
        Var: MetricVar
            variable for a metric
        type: str
            type of plot
        """
        name = self._filenames_lut(type=type)
        ref_meta, mds_meta, other_meta, _, sref_meta = Var.get_varmeta()
        # fetch parts of the name for the variable
        if type in ["barplot_basic", "mapplot_status"]:
            parts = []

        elif type not in ["mapplot_tc", "mapplot_double"]:
            parts = [Var.metric]
            if mds_meta:
                parts.extend([mds_meta[0], mds_meta[1]['short_name']])
        else:
            parts = [ref_meta[0], ref_meta[1]['short_name']]
            if type == "mapplot_tc":
                # necessary to respect old naming convention
                for dss in Var.other_dss:
                    parts.extend([dss[0], dss[1]['short_name']])
                parts.extend(
                    [Var.metric, mds_meta[0], mds_meta[1]['short_name']])
            parts.extend([mds_meta[0], mds_meta[1]['short_name'], Var.metric])

        name = name.format(*parts)
        if period:
            name = f'{period}_{name}'

        return name

    def _yield_values(
        self,
        metric: str,
        tc: bool = False,
        stats: bool = True,
        mean_ci: bool = True,
        short_caption = False
    ) -> Generator[pd.DataFrame, hdl.MetricVariable, pd.DataFrame]:
        """
        Get iterable with pandas dataframes for all variables of a metric to plot

        Parameters
        ----------
        metric: str
            metric name
        tc: bool, default is False
            True if TC. Then, caption starts with "Other Data:"
        stats: bool
            If true, append the statistics to the caption
        mean_ci: bool
            If True, 'Mean CI: {value}' is added to the caption
        short_caption: bool, optional
            whether to use a shorter version of the caption for df columns

        Yield
        -----
        df: pd.DataFrame with variable values and caption name
        Var: QA4SMMetricVariable
            variable corresponding to the dataframe
        ci: pd.DataFrame with "upper" and "lower" CI
        """
        Vars = self.img._iter_vars(type="metric",
                                   filter_parms={"metric": metric})

        if metric in globals.TC_METRICS:
            Vars = filter_out_self_combination_tcmetric_vars(Vars)

        for n, Var in enumerate(Vars):
            values = Var.values[Var.varname]
            # changes if it's a common-type Var
            if Var.g == 'common':
                box_cap_ds = 'All datasets'
            else:
                box_cap_ds = self._box_caption(Var, tc=tc, short_caption=short_caption)
            # setting in global for caption stats
            if globals.boxplot_printnumbers:
                box_cap = '{}'.format(box_cap_ds)
                if stats:
                    box_stats = plm._box_stats(values)
                    box_cap = box_cap + "\n{}".format(box_stats)
            else:
                box_cap = box_cap_ds
            df = values.to_frame(box_cap)
            
            ci = self.img.get_cis(Var)
            
            if ci:  # could be that variable doesn't have CIs - empty list
                ci = pd.concat(ci, axis=1)
                label = ""
                if mean_ci:
                    # get the mean CI range
                    diff = ci["upper"] - ci["lower"]
                    # Problem that nan values were making it impossiple to calculate the mean of the differences
                    # Therefore they were dropped before calculating means
                    # But if only nan-values were calculated no range could be calculated
                    try:
                        ci_range = float(diff.dropna().mean().mean())
                        label = "\nMean CI range: {:.2g}".format(ci_range) if ci_range>0.01 else "\nMean CI Range: <0.01"
                    except:
                        ci_range = "nan"
                        label = "\nMean CI range: nan".format(ci_range)
                df.columns = [df.columns[0] + label]
            else:
                ci = None
            # values are all Nan or NaNf - not plotted
            df_arr = df.to_numpy()
            if np.isnan(df_arr).all() or df_arr.size == 0:
                continue

            yield df, Var, ci

    @staticmethod
    def _get_dataset_dict(Var) -> dict:
        """
        Creates dict containing dataset ids as keys and pretty name (version) as values.
        version only gets appended if there are multiple datasets witth the same pretty name.
        
        Parameters
        ----------
        Var : QA4SMMetricVariable
            Var in the image to make the map for.

        Returns
        -------
        d : dict
            Dict containing id:f"{pretty_name+(version)}" key-value pairs.
        """
        dataset_ref = {Var.Datasets.ref_id:f"{Var.Datasets.ref["pretty_name"]}"}
        dataset_others = {Var.Datasets.others_id[i]:f"{Var.Datasets.others[i]["pretty_name"]}" for i in range(len(Var.Datasets.others))}
        d = dataset_ref | dataset_others
        # Append version number if there are multiple datasets with the same pretty name
        groups = {}
        for k, v in (d).items():
            groups.setdefault(v, []).append(k)

        for i in groups.keys():
            if len(groups[i])>1: 
                for j in groups[i]:
                    d[j] = d[j]+f" ({Var.Datasets.dataset_metadata(j)[1]["pretty_version"]})"
        return d
    
    @staticmethod
    def _get_legend_title(Var) -> str:
        """
        Creates a title for an existing legend from a QA4SMMetricVariable.

        Parameters
        ----------
        Var : QA4SMMetricVariable
            Var in the image to make the map for.

        Returns
        -------
        legend_title : String
            String containing the legend title
        """
        _, _, _, scale_ds, _ = Var.get_varmeta()
        d = QA4SMPlotter._get_dataset_dict(Var)
        
        # Append Unit
        for k in d.keys():
            d[k] = d[k]+f" [{Var.Datasets.dataset_metadata(k)[1]['mu'] if not scale_ds else scale_ds[1]["mu"]}]"
        
        legend_title = "Datasets:\n" + "\n".join(f"{k}: {v}" for k, v in (d).items())
        return legend_title
    
    @staticmethod
    def _append_legend_title(fig, ax, Var) -> tuple:
        """
        Appends a title to an existing legend from a QA4SMMetricVariable.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure that contains the axes and legend.
        ax : matplotlib.axes.Axes
            The subplot axis containing the legend to modify.
        Var : QA4SMMetricVariable
            Variable object used to construct the legend title.

        Returns
        -------
        fig, ax : tuple
            The same figure and axis, with the legend title updated.
        """  
        legend = ax.get_legend()
        legend_title = QA4SMPlotter._get_legend_title(Var)

        # Change Size to same as rest of legend
        if len(legend.legend_handles) == 0:
            legend.set_title(legend_title, prop={'size': globals.fontsize_legend})
        else:
            fs = legend.get_texts()[0].get_fontsize()
            legend.set_title(legend_title, prop={'size': fs})

        legend._legend_box.align = "left"
        best_loc_with_title = plm.best_legend_pos_exclude_list(ax)

        # Step 4: move legend to new best position
        legend.set_loc(best_loc_with_title)

        return fig, ax
    
    @staticmethod
    def _smart_suptitle(fig, pad=globals.fontsize_title/2):
        """
        Compute position of Suptitle centeredd above axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object.
        pad : float
            Extra space (in fontsize) above the top axes title.
        """
        fig.canvas.draw() 

        top_positions = []
        for ax in fig.axes:
            if ax.get_visible():
                # get bounding box of the title in figure coordinates
                title = ax.title
                bbox = title.get_window_extent(renderer=fig.canvas.get_renderer())
                bbox_fig = bbox.transformed(fig.transFigure.inverted())
                top_positions.append(bbox_fig.y1 + title.get_fontsize()/(72*fig.get_figheight()))

        if top_positions:
            y = max(top_positions) + pad/(72*fig.get_figheight())
            y = min(y, 0.99) # So Suptitle always in Figure
        else:
            y = 0.99  # fallback if no axes
        # get center of axes or average of centers if multiple axes
        x = np.mean([(ax.get_position().x0+ax.get_position().x1)/2 for ax in fig.get_axes()[:globals.n_col_agg]])

        return x, y
    
    @staticmethod
    def _smart_suplabel(fig, axis, pad=globals.fontsize_label/2):
        """
        Compute position of suplabels centered according to axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object.
        axis : str
            Axis for which to calculate position.
        pad : float
            Extra space (in fontsize) above the top axes title.
        """
        fig.canvas.draw() 
        if axis == "x":
            bottom_positions = []
            for ax in fig.axes:
                if ax.get_visible():
                    renderer = fig.canvas.get_renderer()

                    # consider x-axis label and tick labels
                    xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=renderer)
                    xtick_bboxes = [t.get_window_extent(renderer=renderer) for t in ax.xaxis.get_ticklabels() if t.get_text()]
                    
                    all_bboxes = [xlabel_bbox] + xtick_bboxes
                    all_bboxes_fig = [b.transformed(fig.transFigure.inverted()) for b in all_bboxes]
                    bottom_positions.append(min(b.y0 for b in all_bboxes_fig))

            if bottom_positions:
                y = min(bottom_positions) - globals.fontsize_label/(72*fig.get_figheight()) - pad/(72*fig.get_figheight())
            else:
                y = 0.01  # fallback
            x = np.mean([(ax.get_position().x0+ax.get_position().x1)/2 for ax in fig.get_axes()[:globals.n_col_agg]])
            return x, y
        elif axis == "y":
            left_positions = []
            for ax in fig.axes:
                if ax.get_visible():
                    renderer = fig.canvas.get_renderer()

                    # consider x-axis label and tick labels
                    ylabel_bbox = ax.yaxis.label.get_window_extent(renderer=renderer)
                    ytick_bboxes = [t.get_window_extent(renderer=renderer) for t in ax.yaxis.get_ticklabels() if t.get_text()]
                    
                    all_bboxes = [ylabel_bbox] + ytick_bboxes
                    all_bboxes_fig = [b.transformed(fig.transFigure.inverted()) for b in all_bboxes]
                    left_positions.append(min(b.x0 for b in all_bboxes_fig))

            if left_positions:
                x = min(left_positions) - globals.fontsize_label/(72*fig.get_figwidth()) - pad/(72*fig.get_figheight()) 
            else:
                x = 0.01  # fallback
            y = np.mean([(ax.get_position().y0+ax.get_position().y1)/2 for ax in fig.get_axes()[::globals.n_col_agg]])
            return x, y
        else: #fallback
            return 0.01, 0.01

    def _get_ax_width(fig) -> float:
        """Get horizontal distance of all axes in px. From left of first in row to right of last in row."""
        left = min([ax.get_position().x0 for ax in fig.get_axes()[:1]]) # Always the first ax
        right = max([ax.get_position().x1 for ax in fig.get_axes()])
        ax_width_px = fig.get_figwidth()*(right-left) * fig.dpi

        return ax_width_px
    
    def _get_ax_height(fig) -> float:
        """Get vertical distance of all axes in px. From bottom of column to top of column."""
        bottom = min([ax.get_position().y0 for ax in fig.get_axes()]) # Always the first ax
        top = max([ax.get_position().y1 for ax in fig.get_axes()])
        ax_height_px = fig.get_figheight()*(top-bottom) * fig.dpi

        return ax_height_px

    @staticmethod
    def _set_wrapped_title(fig, ax, title, fontsize=globals.fontsize_title,
                           pad=globals.title_pad, use_suptitle=False):
        """
        Set an axes or figure suptitle that automatically wraps to fit within figure width.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The target axes (ignored if use_suptitle=True).
        title : str
            The title text.
        fontsize : int
            Font size of the title.
        pad : float
            Extra spacing from the plot, in points.
        use_suptitle : bool, optional
            If True, set a figure-wide suptitle instead of an axes title.
        """
        fig.canvas.draw()  # needed to get renderer

        # width between most left point and most right point in axesin inches * dpi = pixels
        ax_width_px = QA4SMPlotter._get_ax_width(fig)

        # estimate character width (rough, depends on font)
        wrapped = plm.wrapped_text(fig, title, ax_width_px, fontsize)

        if use_suptitle:
            x, y = QA4SMPlotter._smart_suptitle(fig)
            fig.suptitle(wrapped, fontsize=fontsize, y=y, x=x, ha="center", va="bottom")

        else:
            x, y = QA4SMPlotter._smart_suptitle(fig)
            ax.set_title(wrapped, fontsize=fontsize, pad=pad)

    def _boxplot_definition(self,
                            metric: str,
                            df: pd.DataFrame,
                            type: str,
                            period: str = None,
                            ci=None,
                            offset=0.07,
                            Var=None,
                            **kwargs) -> tuple:
        """
        Define parameters of plot

        Parameters
        ----------
        df: pd.DataFrame to plot
        type: str
            one of _titles_lut
        ci : list of Dataframes containing "upper" and "lower" CIs
        xticks: list
            caption to each boxplot (or triplet thereof)
        offset: float
            offset of boxplots
        Var: QA4SMMetricVariable, optional. Default is None
            Specified in case mds meta is needed
        """
        # plot label
        unit_ref = self.ref['short_name']

        _, _, _, scl_meta, sref_meta = Var.get_varmeta()
        parts = [globals._metric_name[metric]]
        label = "{}".format(*parts)
        figsize = [globals.boxplot_width_vertical, globals.boxplot_height_vertical]
        fig, ax = plm.boxplot(
            df=df,
            ci=ci,
            label=label,
            figsize=figsize,
        )
        if not Var:
            # when we only need reference dataset from variables (i.e. is the same):
            for Var in self.img._iter_vars(type="metric",
                                           filter_parms={"metric": metric}):
                Var = Var
                break
        if not type == "metadata":
            title = self.create_title(Var, type=type, period=period)
            QA4SMPlotter._set_wrapped_title(fig, ax[0], title)

        # Legendentries for Datasets
        fig, ax[0] = QA4SMPlotter._append_legend_title(fig, ax[0], Var)

        if globals.draw_logo:
            plm.add_logo_to_figure(
                fig=fig,
                logo_path=globals.logo_pth,
                position=globals.logo_position,
                offset=globals.logo_offset_box_plots,
                size=globals.logo_size
            )

        return fig, ax

    def _barplot_definition(self,
                            metric: str,
                            df: pd.DataFrame,
                            type: str,
                            period: str = None,
                            Var=None) -> tuple:
        """
        Define parameters of plot

        Parameters
        ----------
        df: pd.DataFrame to plot
        type: str
            one of _titles_lut
        metric : str
            metric that is collected from the file for all variables.
        Var: QA4SMMetricVariable, optional. Default is None
            Specified in case mds meta is needed
        """
        # plot label
        parts = [globals._metric_name[metric]]
        label = "{}".format(*parts)
        # otherwise it's too narrow
        figsize = [globals.boxplot_width_vertical, globals.boxplot_height_vertical]
        fig, ax = plm.barplot(df=df,
                              figsize=figsize,
                              label='# of validation errors')
        if not Var:
            # when we only need reference dataset from variables (i.e. is the same):
            for Var in self.img._iter_vars(type="metric",
                                           filter_parms={"metric": metric}):
                Var = Var
                break

        title = self.create_title(Var, type=type, period=period)
        QA4SMPlotter._set_wrapped_title(fig, ax, title)

        # add logo
        if globals.draw_logo:
            plm.add_logo_to_figure(
                fig=fig,
                logo_path=globals.logo_pth,
                position=globals.logo_position,
                offset=globals.logo_offset_box_plots,
                size=globals.logo_size
            )

    def _save_plot(self,
                   out_name: str,
                   out_types: Optional[Union[List[str], str]] = 'png') -> list:
        """
        Save plot with name to self.out_dir

        Parameters
        ----------
        out_name: str
            name of output file
        out_types: str or list of str, Optional
            extensions which the files should be saved in. Default is 'png'

        Returns
        -------
        fnames: list of file names with all the extensions
        """
        fnames = []
        if isinstance(out_types, str):
            out_types = [out_types]
        for ext in out_types:
            fname = self._standard_filename(out_name, out_type=ext)
            if fname.exists():
                warn(f'Overwriting file {fname.name}')
            try:
                plt.savefig(fname, dpi='figure', bbox_inches='tight')
            except ValueError:
                continue
            fnames.append(fname.absolute())

        return fnames

    def boxplot_basic(self,
                      metric: str,
                      period: str = None,
                      out_name: str = None,
                      out_types: Optional[Union[List[str], str]] = 'png',
                      save_files: bool = False,
                      **plotting_kwargs) -> Union[list, None]:
        """
        Creates a boxplot for common and double metrics. Saves a figure and
        returns Matplotlib fig and ax objects for further processing.

        Parameters
        ----------
        metric : str
            metric that is collected from the file for all datasets and combined
            into one plot.
        out_name: str
            name of output file
        out_types: str or list of str, Optional
            extensions which the files should be saved in. Default is 'png'
        save_files: bool, optional. Default is False
            wether to save the file in the output directory
        plotting_kwargs: arguments for _boxplot_definition function

        Returns
        -------
        fnames: list of file names with all the extensions
        """
        fnames, values = [], []
        ci = {}
        # we take the last iterated value for Var and use it for the file name
        for df, Var, var_ci in self._yield_values(metric=metric, short_caption=False):
            values.append(df)
            if var_ci is not None:
                ci[f"{Var.ref_ds[0]} & {Var.metric_ds[0]}"] = var_ci

        # handle empty results
        if not values:
            return None
        # put all Variables in the same dataframe
        values = pd.concat(values, axis=1)
        values = values.reset_index().melt(id_vars = ["lat", "lon", "gpi"], var_name = "label", value_name="value").sort_values("label")
        values["dataset"] = [values["label"][i].split("\n")[0].replace("Datasets: ", "") for i in values.index]

        # create plot
        fig, ax = self._boxplot_definition(metric=metric,
                                           df=values,
                                           type='boxplot_basic',
                                           period=period,
                                           ci=ci,
                                           Var=Var,
                                           **plotting_kwargs)
        if not out_name:
            out_name = self.create_filename(Var,
                                            type='boxplot_basic',
                                            period=period)
        # save or return plotting objects
        if save_files:
            fnames.extend(self._save_plot(out_name, out_types=out_types))
            plt.close('all')

            return fnames

        else:
            return fig, ax
        
    def boxplot_tc(self,
                   metric: str,
                   period: str = None,
                   out_name: str = None,
                   out_types: Optional[Union[List[str], str]] = 'png',
                   save_files: bool = False,
                   **plotting_kwargs) -> list:
        """
        Creates a boxplot for TC metrics. Saves a figure and returns Matplotlib fig and ax objects for
        further processing.

        Parameters
        ----------
        metric : str
            metric that is collected from the file for all datasets and combined
            into one plot.
        out_name: str
            name of output file
        out_types: str or list of str, Optional
            extensions which the files should be saved in. Default is 'png'
        save_files: bool, optional. Default is False
            wether to save the file in the output directory
        plotting_kwargs: arguments for _boxplot_definition function

        Returns
        -------
        fnames: list of file names with all the extensions
        """
        fnames = []
        # group Vars and CIs relative to the same dataset
        metric_tc, ci = {}, {}

        for df, Var, var_ci in self._yield_values(metric=metric, tc=True):
            id = Var.metric_ds[0]
            ref_ds, metric_ds, other_ds, _, _ = Var.get_varmeta()
            if var_ci is not None:
                # var_ci["dataset"] = [f"{ref_ds[0]} & {other_ds[0]}" for i in range(len(var_ci))]
                if id in ci.keys():
                    ci[id][f"{ref_ds[0]} & {other_ds[0]}"]=var_ci
                else:
                    ci[id] = {f"{ref_ds[0]} & {other_ds[0]}":var_ci}
            if id in metric_tc.keys():
                metric_tc[id][0].append(df)
            else:
                metric_tc[id] = [df], Var

        for id, values in metric_tc.items():
            dfs, Var = values
            df = pd.concat(dfs)
            df = df.reset_index().melt(id_vars = ["lat", "lon", "gpi"], var_name = "label", value_name="value").sort_values("label")
            # df["dataset"] = f"{Var.ref_ds[0]} & {Var.metric_ds[0]}" 
            df["dataset"] = [df["label"][i].split("\n")[1].replace("Datasets: ", "") for i in df.index] 
            # Because the plots have to be generated in comparions with each pair of other data in tc so ifthere are 5 datasets i calculate the tc for 1 with 0-2, 0-3, 0-4
            # values are all Nan or NaNf - not plotted
            if np.isnan(df["value"].to_numpy()).all():
                continue
            # necessary if statement to prevent key error when no CIs are in the netCDF
            if ci:
                ci_id = ci[id]
            else:
                ci_id = None
            # create plot
            fig, ax = self._boxplot_definition(metric=metric,
                                               df=df,
                                               ci=ci_id,
                                               type='boxplot_tc',
                                               period=period,
                                               Var=Var,
                                               **plotting_kwargs)
            # save. Below workaround to avoid same names
            if not out_name:
                save_name = self.create_filename(Var,
                                                 type='boxplot_tc',
                                                 period=period)
            else:
                save_name = out_name
            # save or return plotting objects
            if save_files:
                fnames.extend(self._save_plot(save_name, out_types=out_types))
                plt.close('all')

        if save_files:
            return fnames

    def barplot(
        self,
        metric: str,
        period: str = None,
        out_types: Optional[Union[List[str], str]] = 'png',
        save_files: bool = False,
    ) -> Union[list, None]:
        """
        Creates a barplot of validation errors betweeen two or three datasets.
        Saves a figure and returns Matplotlib fig and ax objects for
        further processing.

        Parameters
        ----------
        metric : str
            metric that is collected from the file for all datasets.
        out_types: str or list of str, Optional
            extensions which the files should be saved in. Default is 'png'
        save_files: bool, optional. Default is False
            wether to save the file in the output directory

        Returns
        -------
        fnames: list of file names with all the extensions
        """
        fnames, values = [], []
        # we take the last iterated value for Var and use it for the file name
        for values, Var, _ in self._yield_values(metric=metric):
            # handle empty results
            if values.empty:
                return None

            if len(self.img.triple) and Var.g == 'pairwise':
                continue

            ref_meta, mds_meta, other_meta, _, sref_meta = Var.get_varmeta()

            self._barplot_definition(metric=metric,
                                     df=values,
                                     type='barplot_basic',
                                     period=period,
                                     Var=Var)

            out_name = self.create_filename(Var,
                                            type='barplot_basic',
                                            period=period)
            # save or return plotting objects
            if save_files:
                fnames.extend(self._save_plot(out_name, out_types=out_types))
                plt.close('all')

        if fnames:
            return fnames

    def mapplot_var(
        self,
        Var,
        period: str = None,
        out_types: Optional[Union[List[str], str]] = 'png',
        save_files: bool = False,
        compute_dpi: bool = True,
        **style_kwargs,
    ) -> Union[list, Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]:
        """
        Plots values to a map, using the values as color. Plots a scatterplot for
        ISMN and a image plot for other input values.

        Parameters
        ----------
        Var : QA4SMMetricVariable
            Var in the image to make the map for.
        out_name: str
            name of output file
        out_types: str or list of str, Optional
            extensions which the files should be saved in. Default is 'png'
        save_files: bool, optional. Default is False
            wether to save the file in the output directory
        compute_dpi : bool, optional. Default is True.
            if selected, the output resolution of the image is
            calculated on the basis of the resolution of the
            reference dataset and the extent of the validation
            (i.e., high resolution, global validations will
            have the maximum available dpi).
            Otherwise, high resolution datasets are assigned the
            maximum dpi as per globals.max_dpi
        style_kwargs: arguments for mapplot function

        Returns
        -------
        fnames: list of file names with all the extensions
        """
        fnames = []
        ref_meta, mds_meta, other_meta, scl_meta, sref_meta = Var.get_varmeta()
        metric = Var.metric
        ref_grid_stepsize = self.img.ref_dataset_grid_stepsize
        res, unit = self.img.res_info.values()
        extent = self.img.extent

        if res is not None:
            if compute_dpi and extent is not None:
                dpi = plm.output_dpi(
                    res,
                    unit,
                    extent,
                )

                style_kwargs["dpi"] = dpi

            elif not compute_dpi and unit == "km" and res <= 1:
                style_kwargs["dpi"] = globals.dpi_max

        # get the short_name of the scaling reference
        scl_short = None
        if scl_meta:
            scl_short = scl_meta[1]['short_name']

        is_scattered = Var.attrs.get('val_is_scattered_data') == 'True'
        # create mapplot
        fig, ax = plm.mapplot(
            df=Var.values[Var.varname],
            metric=metric,
            ref_short=sref_meta[1]['short_name'],
            ref_grid_stepsize=ref_grid_stepsize,
            plot_extent=
            None,  # if None, extent is automatically adjusted (as opposed to img.extent)
            scl_short=scl_short,
            is_scattered=is_scattered,
            **style_kwargs)

        # title and plot settings depend on the metric group
        if Var.varname.startswith('status'):
            title = self.create_title(Var=Var,
                                      type='mapplot_status',
                                      period=period)
            save_name = self.create_filename(Var=Var,
                                             type="mapplot_status",
                                             period=period)
        elif Var.g == 'common':
            title = "{} between all datasets".format(
                globals._metric_name[metric])
            if period:
                if period not in globals.no_print_period:
                    title = f'{period}: {title}'
            save_name = self.create_filename(Var,
                                             type='mapplot_common',
                                             period=period)
        elif Var.g == 'pairwise' or Var.g == 'pairwise_stability':
            title = self.create_title(Var=Var,
                                      type='mapplot_basic',
                                      period=period)
            save_name = self.create_filename(Var,
                                             type='mapplot_double',
                                             period=period)
        else:
            title = self.create_title(Var=Var,
                                      type='mapplot_tc',
                                      period=period)
            save_name = self.create_filename(Var,
                                             type='mapplot_tc',
                                             period=period)

        # use title for plot, make logo
        QA4SMPlotter._set_wrapped_title(fig, ax, title)

        if globals.draw_logo:
            plm.add_logo_to_figure(
                fig=fig,
                logo_path=globals.logo_pth,
                position=globals.logo_position,
                size=globals.logo_size
            )

        # save file or just return the image
        if save_files:
            fnames.extend(self._save_plot(save_name, out_types=out_types))
            plt.close('all')
            return fnames

        else:
            return fig, ax

    def mapplot_metric(self,
                       metric: str,
                       period: str = None,
                       out_types: Optional[Union[List[str], str]] = 'png',
                       save_files: bool = False,
                       **plotting_kwargs) -> list:
        """
        Mapplot for all variables for a given metric in the loaded file.

        Parameters
        ----------
        metric : str
            Name of a metric. File is searched for variables for that metric.
        out_types: str or list of str, Optional
            extensions which the files should be saved in. Default is 'png'
        save_files: bool, optional. Default is False
            wether to save the file in the output directory
        plotting_kwargs: arguments for mapplot function

        Returns
        -------
        fnames : list
            List of files that were created
        """
        fnames = []
        for Var in self.img._iter_vars(type="metric",
                                       filter_parms={"metric": metric}):
            if len(self.img.triple
                   ) and Var.g == 'pairwise' and metric == 'status':
                continue
            if not (np.isnan(Var.values.to_numpy()).all() or Var.is_CI):
                fns = self.mapplot_var(Var,
                                       period=period,
                                       out_types=out_types,
                                       save_files=save_files,
                                       **plotting_kwargs)
            # values are all Nan or NaNf - not plotted
            else:
                continue
            if save_files:
                fnames.extend(fns)
                plt.close('all')

        if fnames:
            return fnames

    def plot_metric(self,
                    metric: str,
                    period: str = None,
                    out_types: Optional[Union[List[str], str]] = 'png',
                    save_all: bool = True,
                    **plotting_kwargs) -> tuple:
        """
        Plot and save boxplot and mapplot for a certain metric

        Parameters
        ----------
        metric: str
            name of the metric
        out_types: str or list
            extensions which the files should be saved in
        save_all: bool, optional. Default is True.
            all plotted images are saved to the output directory
        plotting_kwargs: arguments for mapplot function.
        """
        fnames_bplot = None
        Metric = self.img.metrics[metric]

        fnames_mapplot = None
        if Metric.name == 'status':
            fnames_bplot = self.barplot(metric='status',
                                        period=period,
                                        out_types=out_types,
                                        save_files=save_all)

        elif Metric.g == 'common' or Metric.g == 'pairwise' or Metric.g == 'pairwise_stability':
            fnames_bplot = self.boxplot_basic(metric=metric,
                                              period=period,
                                              out_types=out_types,
                                              save_files=save_all,
                                              **plotting_kwargs)
        elif Metric.g == 'triple':
            fnames_bplot = self.boxplot_tc(metric=metric,
                                           period=period,
                                           out_types=out_types,
                                           save_files=save_all,
                                           **plotting_kwargs)
        if period == globals.DEFAULT_TSW:
            fnames_mapplot = self.mapplot_metric(metric=metric,
                                                 period=period,
                                                 out_types=out_types,
                                                 save_files=save_all,
                                                 **plotting_kwargs)

        return fnames_bplot, fnames_mapplot

    def meta_single(self,
                    metric: str,
                    metadata: str,
                    df: pd.DataFrame = None,
                    axis=None,
                    plot_type: str = "catplot",
                    **plotting_kwargs) -> Union[tuple, None]:
        """
        Boxplot of a metric grouped by the given metadata.

        Parameters
        ----------
        metric : str
            specified metric
        metadata : str
            specified metadata
        df : pd.DataFrame, optional
            metric values can be specified, in which case they will be used from here and
            not parsed from the metric name
        axis : matplotlib.axes.Axis, optional
            if provided, the function will create the plot on the specified axis
        plot_type : str, default is 'catplot'
            one of 'catplot' or 'multiplot', defines the type of plots for the 'classes' and 'continuous'
            metadata types
        plotting_kwargs:
            Keyword arguments for the plotting function


        Returns
        -------
        fig : matplotlib.figure.Figure
            the boxplot
        ax : matplotlib.axes.Axes
        """

        values = []
        for data, Var, var_ci in self._yield_values(metric=metric,
                                                    stats=False,
                                                    mean_ci=False,
                                                    short_caption=True):
            values.append(data)
        if not values:
            raise PlotterError(f"No valid values for {metric}")
        values = pd.concat(values, axis=1)

        # override values from metric
        if df is not None:
            values = df

        # get meta and select only metric values with metadata available
        meta_values = self.img.metadata[metadata].values.dropna()
        values = values.reindex(index=meta_values.index)

        # For tca with > 3 datasets, it can be that the same metric comes
        # from different combinations of datasets, in that case take average
        # the different versions
        for col in np.unique(values.columns):
            loc = np.where(values.columns == col)[0]
            if len(loc) == 1:
                continue
            else:
                colmean = values.pop(col).mean(axis=1, skipna=True)
                values[col] = colmean

        out = plm.boxplot_metadata(df=values,
                                   metadata_values=meta_values,
                                   ax_label=Var.Metric.pretty_name,
                                   axis=axis,
                                   plot_type=plot_type,
                                   **plotting_kwargs)
        
        if out:
            out = QA4SMPlotter._append_legend_title(out[0], out[1], Var)

        if axis is None:
            fig, ax = out

            return fig, ax

    def meta_combo(
        self,
        metric: str,
        metadata: str,
        metadata_discrete: str,
        **plotting_kwargs,
    ):
        """
        Cross-boxplot between two given metadata types

        Parameters
        ----------
        metric : str
            specified metric
        metadata: str
            'continuous' or 'classes' metadata which provides the number of subplots (bins)
        metadata_discrete : str
            'discrete' metadata which is shown in the subplots

        Returns
        -------
        fig : matplotlib.figure.Figure
            the boxplot
        ax : matplotlib.axes.Axes
        """
        values = []
        for df, Var, ci in self._yield_values(metric=metric,
                                              stats=False,
                                              mean_ci=False,
                                              short_caption=True):
            values.append(df)
        if not values:
            raise PlotterError(f"No valid values for {metric}")
        values = pd.concat(values, axis=1)

        metric_name = globals._metric_name[metric]

        Meta_cont = self.img.metadata[metadata]
        meta_values = Meta_cont.values.dropna()

        bin_funct = plm.bin_function_lut(globals.metadata[metadata][2])
        kwargs = dict()
        if 'meta_boxplot_min_samples' in plotting_kwargs:
            kwargs['min_size'] = plotting_kwargs['meta_boxplot_min_samples']

        binned_values = bin_funct(df=values,
                                  metadata_values=meta_values,
                                  meta_key=metadata,
                                  **kwargs)
        if binned_values is None:
            raise PlotterError(
                f"Could not bin metadata {metadata} with function {bin_funct}")

        values_subset = {
            a_bin: values.reindex(index=binned_values[a_bin].index)
            for a_bin in binned_values.keys()
        }
        kwargs = {
            "metric": metric,
            "metadata": metadata_discrete,
            "n_bins": len(set(self.img.metadata[metadata_discrete].values.dropna()[self.img.metadata[metadata_discrete].values.dropna().keys()[0]]))}

        plotting_kwargs.update(kwargs)
        fig, axes = plm.aggregate_subplots(to_plot=values_subset,
                                           funct=self.meta_single,
                                           **plotting_kwargs)
        
        # Append Legend Title
        if isinstance(axes, plt.Axes):   # single axes case
            ax_first = axes
        else:
            ax_first = axes.flat[0]

        ax_first.legend(fontsize=globals.fontsize_legend).set_loc("upper left")
        _, ax_first = QA4SMPlotter._append_legend_title(fig, ax_first, Var)
        
        return fig, axes

    def plot_metadata(self,
                      metric: str,
                      metadata: str,
                      metadata_discrete: str = None,
                      save_file: bool = False,
                      out_types: Optional[Union[List[str], str]] = 'png',
                      period: str = None,
                      **plotting_kwargs):
        """
        Wrapper built around the 'meta_single' or 'meta_combo' functions to produce a metadata-based boxplot of a
        metric.

        Parameters
        ----------
        metric : str
            name of metric to plot
        metadata : str
            name of metadata to subdivide the metric results
        metadata_discrete : str
            name of the metadata of the type 'discrete'
        save_file : bool, optional
            whether to save the plot to the output directory. Default is False
        out_types : str or list of str, optional
            extensions which the files should be saved in. Default is 'png'
        period: str, optional
            temporal sub-window to use

        Retrun
        ------
        fig : matplotlib.figure.Figure
            the boxplot
        ax : matplotlib.axes.Axes
        """
        fnames = []
        if metadata_discrete is None:
            fig, ax = self.meta_single(metric=metric,
                                       metadata=metadata,
                                       **plotting_kwargs)
            metadata_tuple = [metadata]

        else:
            metadata_tuple = [metadata, metadata_discrete]
            if not any(globals.metadata[i][2] == "discrete"
                       for i in metadata_tuple):
                raise ValueError(
                    "One of the provided metadata names should correspond to the 'discrete' type, see globals.py"
                )
            if all(globals.metadata[i][2] == "discrete"
                   for i in metadata_tuple):
                raise ValueError(
                    "At least one of the provided metadata should not be of the 'continuous' type"
                )
            fig, ax = self.meta_combo(metric=metric,
                                      metadata=metadata,
                                      metadata_discrete=metadata_discrete,
                                      **plotting_kwargs)
        meta_names = [globals.metadata[i][0] for i in metadata_tuple]
        title = self._titles_lut("metadata").format(
            globals._metric_name[metric], ", ".join(meta_names),
            self.img.datasets.ref["pretty_title"])
        
        # Appending labels for metadata Axis
        if isinstance(ax, plt.Axes):   # Only for single metadata plots, distracting in multiple subplots
            if ax.get_ylabel() in [""]+metadata_tuple:
                ax_width_px = QA4SMPlotter._get_ax_width(fig)
                ylabel = plm.wrapped_text(fig, "metadata: " + globals.metadata[metadata_discrete][0] if metadata_discrete else "metadata: " + ", ".join(meta_names), ax_width_px, globals.fontsize_label)
                ax.set_ylabel(ylabel, fontsize = globals.fontsize_label)
            elif ax.get_xlabel() == [""]+metadata_tuple:
                ax_height_px = QA4SMPlotter._get_ax_height(fig)
                xlabel = plm.wrapped_text(fig, "metadata: " + globals.metadata[metadata_discrete][0] if metadata_discrete else "metadata: " + ", ".join(meta_names), ax_height_px, globals.fontsize_label)
                ax.set_xlabel(xlabel, fontsize = globals.fontsize_label)
        
        elif isinstance(ax, np.ndarray):
            for axis in ax.flat:
                axis.set_ylabel, axis.set_xlabel = "", ""

            if len(ax)==1:
                if fig.get_supylabel() in [""]+metadata_tuple:
                    x, y = QA4SMPlotter._smart_suplabel(fig, "y")
                    ax_width_px = QA4SMPlotter._get_ax_width(fig)
                    ylabel = plm.wrapped_text(fig, "metadata: " + globals.metadata[metadata_discrete][0] if metadata_discrete else "metadata: " + ", ".join(meta_names), ax_width_px, globals.fontsize_label)
                    fig.supylabel(ylabel, 
                                fontsize=globals.fontsize_label, 
                                y=y,
                                x=x)
                if fig.get_supxlabel() in [""]+metadata_tuple:
                    x, y = QA4SMPlotter._smart_suplabel(fig, "x")
                    ax_height_px = QA4SMPlotter._get_ax_height(fig)
                    xlabel = plm.wrapped_text(fig, globals._metric_name[metric], ax_height_px, globals.fontsize_label)
                    fig.supxlabel(xlabel, 
                                fontsize=globals.fontsize_label, 
                                y=y, 
                                x=x)   
            else:
                if fig.get_supylabel() in [""]+metadata_tuple:
                    x, y = QA4SMPlotter._smart_suplabel(fig, "y")
                    ax_width_px = QA4SMPlotter._get_ax_width(fig)
                    ylabel = plm.wrapped_text(fig, globals._metric_name[metric], ax_width_px, globals.fontsize_label)
                    fig.supylabel(ylabel, 
                                fontsize=globals.fontsize_label, 
                                y=y,
                                x=x)
                if fig.get_supxlabel() in [""]+metadata_tuple:
                    x, y = QA4SMPlotter._smart_suplabel(fig, "x")
                    ax_height_px = QA4SMPlotter._get_ax_height(fig)
                    xlabel = plm.wrapped_text(fig, "metadata: " + globals.metadata[metadata_discrete][0] if metadata_discrete else "metadata: " + ", ".join(meta_names), ax_height_px, globals.fontsize_label)
                    fig.supxlabel(xlabel, 
                                fontsize=globals.fontsize_label, 
                                y=y, 
                                x=x)       

        if period:  #$$
            if period not in globals.no_print_period:
                title = f'{period}: {title}'

        QA4SMPlotter._set_wrapped_title(fig, None, title, use_suptitle=True)
        
        if globals.draw_logo:
            plm.add_logo_to_figure(
                fig=fig,
                logo_path=globals.logo_pth,
                position=globals.logo_position,
                offset=globals.logo_offset_metadata_plots,
                size=globals.logo_size,
            )

        if save_file:
            out_name = self._filenames_lut("metadata").format(
                metric, "_and_".join(metadata_tuple))
            if period:
                out_name = f'{period}_{out_name}'
            fnames.extend(self._save_plot(out_name, out_types=out_types))
            plt.close('all')
            return fnames

        else:
            return fig, ax

    def plot_save_metadata(
        self,
        metric,
        out_types: Optional[Union[List[str], str]] = 'png',
        meta_boxplot_min_samples: int = 5,
        period: str = None,
    ):
        """
        Plots and saves three metadata boxplots per metric (defined in globals.py):

        1. Boxplot by land cover class (2010 map)
        2. Boxplot by Koeppen-Geiger climate classification
        3. Boxplot by instrument depth and soil type (granularity)

        Parameters
        ----------
        metric : str
            name of metric
        out_types: str or list of str, optional
            extensions which the files should be saved in. Default is 'png'
        meta_boxplot_min_samples: int, optional
            minimum number of samples per bin required to plot a metadata boxplot

        Return
        ------
        filenames: list
            list of file names
        """
        filenames = []

        # makes no sense to plot the metadata for some metrics
        if metric in globals._metadata_exclude:
            return filenames

        if not period:  #$$
            return filenames

        for meta_type, meta_keys in globals.out_metadata_plots.items():
            try:
                # the presence of instrument_depth in the out file depends on the ismn release version
                if all(meta_key in self.img.metadata.keys()
                       for meta_key in meta_keys):
                    outfiles = self.plot_metadata(
                        metric,
                        *meta_keys,
                        save_file=True,
                        out_types=out_types,
                        meta_boxplot_min_samples=meta_boxplot_min_samples,
                        period=period,
                    )
                    filenames.extend(outfiles)

                else:
                    warnings.warn("Not all: " + ", ".join(meta_keys) +
                                  " are present in the netCDF variables")
            except PlotterError as e:
                warnings.warn(
                    f"Too few points are available to generate '{meta_type}` "
                    f"metadata-based `{metric}` plots.")

        return filenames

    def save_stats(self, period: str = None) -> str:
        """Saves the summary statistics to a .csv file and returns the name"""
        table = self.img.stats_df()
        filename = self._filenames_lut("table") + '.csv'
        if period:
            filename = f'{period}_{filename}'
        filepath = self.out_dir.joinpath(filename)
        table.to_csv(path_or_buf=filepath)

        return filepath


#$$
class QA4SMCompPlotter:
    """
    Class to create plots containing the calculated metric for all temporal sub-window, default case excldued

    Parameters
    ----------
    results_file : str
        path to the .nc results file
    include_default_case : bool, default is False
        whether to include the bulk case in the plots

    Returns
    -------
    QA4SMCompPlotter object
    """

    def __init__(self,
                 results_file: str,
                 include_default_case: bool = False) -> None:
        self.results_file = results_file
        self.include_default_case = include_default_case
        if os.path.isfile(results_file):
            with xr.open_dataset(results_file) as ds:
                self.ds: xr.Dataset = ds
                self.datasets = hdl.QA4SMDatasets(self.ds.attrs)
                self.ref_dataset: Dict = self.datasets.ref
                self.candidate_datasets: List[Dict] = self.datasets.others
                # self.metrics_in_ds = self.__ds_metrics()
                self.metric_kinds_available: List = list(
                    self.metrics_in_ds.keys())
                self.metric_lut: Dict = self.metrics_ds_grouped_lut(
                    include_ci=False)
                # self.df = self._ds2df()
                # self.check_for_unexpecetd_metrics()

                self.cbp: ClusteredBoxPlot = ClusteredBoxPlot(
                    anchor_list=np.linspace(1, len(self.tsws_used),
                                            len(self.tsws_used)),
                    no_of_ds=len(self.candidate_datasets),
                    space_per_box_cluster=0.9,
                    rel_indiv_box_width=0.9,
                )

        else:
            warnings.warn(
                f'FileNotFoundError: The file {results_file} does not exist. Please check the file path and try again.'
            )
            return None

    @property
    def temp_sub_win_dependent_vars(self) -> List[str]:
        _list = []
        for var_name in self.ds.data_vars:
            if 'tsw' in self.ds[var_name].dims:
                _list.append(var_name)
        return _list

    @property
    def metrics_in_ds(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary of metrics in the dataset, whereas each individual metric kind is a key in the dictionary \
        and the values are lists of variables in the dataset that are associated with the respective metric kind.

        Returns
        -------
        dict
            dictionary of metrics in the dataset

        """
        return {
            metric: [
                var_name for var_name in self.temp_sub_win_dependent_vars
                if var_name.startswith(f'{metric}_')
            ]
            for metric in globals._colormaps.keys() if any(
                var_name.startswith(f'{metric}_')
                for var_name in self.temp_sub_win_dependent_vars)
        }  #  globals._colormaps just so happens to contain all metrics

    def check_for_unexpecetd_metrics(self) -> bool:
        """
        Checks if the metrics are present in the dataset that were not specified in `globals.METRICS` and adds them to \
        `QA4SMCompPlotter.ds_metrics`.

        Returns
        -------
        bool
            True if no unexpected metrics are found in the dataset, False otherwise
        """

        flattened_list = list(
            set(itertools.chain.from_iterable(self.metrics_in_ds.values())))
        elements_not_in_flattened_list = set(
            self.temp_sub_win_dependent_vars) - set(flattened_list)
        _list = list(
            set([
                m.split('_between')[0] for m in elements_not_in_flattened_list
            ]))
        grouped_dict = {
            prefix:
            [element for element in _list if element.startswith(prefix)]
            for prefix in set([element.split('_')[0] for element in _list])
        }

        for prefix, elements in grouped_dict.items():
            self.metrics_in_ds[prefix] = elements

        if len(elements_not_in_flattened_list) > 0:
            warnings.warn(
                f"Following metrics were found in the dataset that were not specified in `globals.METRICS` and have \
                been added to `QA4SMCompPlotter.ds_metrics`: {elements_not_in_flattened_list}"
            )
            return False

        return True

    def metrics_ds_grouped_lut(self,
                               include_ci: Optional[bool] = False
                               ) -> Dict[str, List[str]]:
        """
        Returns a dictionary of for each metric, containing the QA4SM dataset combination used to compute said metric

        Parameters
        ----------
        include_ci : bool, default is False
            Whether to include the confidence intervals of a specific metric in the output

        Returns
        -------
        dict
            dictionary of grouped metrics in the dataset
        """
        _metric_lut = {}

        def parse_metric_string(
                metric_string: str) -> Union[Tuple[str, str], None]:
            pattern = globals.METRIC_TEMPLATE.format(
                ds1=
                r'(?P<ds1>\d+-\w+)',  # matches one or more digits (\d+), followed by a hyphen (-), \
                # followed by one or more word characters (\w+)
                ds2=
                r'(?P<ds2>\d+-\w+)',  # matches one or more digits (\d+), followed by a hyphen (-), \
                # followed by one or more word characters (\w+)
            )

            match = re.search(pattern, metric_string)
            if match:
                return match.group('ds1'), match.group('ds2')
            else:
                return None

        def purge_ci_metrics(_dict: Dict) -> Dict:
            return {
                ds_combi:
                [metric for metric in metric_values if "ci" not in metric][0]
                for ds_combi, metric_values in _dict.items()
            }

        for metric_kind, metrics_in_ds in self.metrics_in_ds.items():

            parsed_metrics = set([
                pp for metric in metrics_in_ds
                if (pp := parse_metric_string(metric)) is not None
            ])

            grouped_dict = {
                metric: [
                    item for item in metrics_in_ds
                    if parse_metric_string(item) == metric
                ]
                for metric in parsed_metrics
            }

            if not include_ci:
                grouped_dict = purge_ci_metrics(grouped_dict)

            _metric_lut[metric_kind] = grouped_dict

        return _metric_lut

    @property
    def tsws_used(self):
        """
        Get all temporal sub-windows used in the validation

        Parameters
        ----------
        incl_default : bool, default is False
            Whether to include the default TSW in the output


        Returns
        -------
        tsws_used : list
            list of all TSWs used in the validation
        """

        temp_sub_wins_names = [
            tsw for tsw in self.ds.coords[
                globals.TEMPORAL_SUB_WINDOW_NC_COORD_NAME].values
            if tsw != globals.DEFAULT_TSW
        ]

        if self.include_default_case:
            temp_sub_wins_names.append(globals.DEFAULT_TSW)

        return temp_sub_wins_names

    def get_specific_metric_df(self, specific_metric: str) -> pd.DataFrame:
        """
        Get the DataFrame for a single **specific** metric (e.g. "R_between_0-ISMN_and_1-SMOS_L3") from a QA4SM netCDF \
        file with temporal sub-windows.

        Parameters
        ----------
        specific_metric : str
            Name of the specific metric

        Returns
        -------
        pd.DataFrame
            DataFrame for this specific metric
        """

        _data_dict = {}
        _data_dict['lat'] = self.ds['lat'].values
        _data_dict['lon'] = self.ds['lon'].values
        _data_dict['gpi'] = self.ds['gpi'].values
        for tsw in self.tsws_used:
            selection = {globals.TEMPORAL_SUB_WINDOW_NC_COORD_NAME: tsw}

            _data_dict[tsw] = self.ds[specific_metric].sel(
                selection).values.astype(np.float32)

        df = pd.DataFrame(_data_dict)
        df.set_index(['lat', 'lon', 'gpi'], inplace=True)

        return df

    def get_metric_df(self, generic_metric: str) -> pd.DataFrame:
        """
        Get the DataFrame for a single **generic** metric/metric kind (e.g. "R") from a QA4SM netCDF file with \
        temporal sub-windows.

        Parameters
        ----------
        generic_metric : str
            Name of the generic metric/metric kind

        Returns
        -------
        pd.DataFrame
            Multilevel DataFrame for this generic metric/metric kind, whereas the two column levels are all candidate \
            datasets and the temporal sub-windows
        """

        df_dict = {
            ds_combi:
            self.get_specific_metric_df(specific_metric=specific_metric)
            for ds_combi, specific_metric in
            self.metric_lut[generic_metric].items()
        }
        return pd.concat(df_dict.values(), keys=df_dict.keys(), axis=1)

    @staticmethod
    @note(
        "This method is redundant, as it yields the same result as `QA4SMCompPlotter.tsws_used()`. \
        It is kept as a static method for debugging purposes.")
    def get_tsws_from_df(df: pd.DataFrame) -> List[str]:
        """
        Get all temporal sub-windows used in the validation from a DataFrame as returned by \
        `QA4SMCompPlotter.get_metric_df()`

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the temporal sub-windows

        Returns
        -------
        tsws_used : list
            list of all TSWs used in the validation
        """
        return df.columns.levels[1].unique().tolist()

    @staticmethod
    def get_datasets_from_df(df: pd.DataFrame) -> List[str]:
        """
        Get all candiate datasets used in the validation from a DataFrame as returned by \
        `QA4SMCompPlotter.get_metric_df()`

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the datasets

        Returns
        -------
        datasets_used : list
            list of all datasets used in the validation
        """
        return sorted(df.columns.levels[0].unique().tolist())

    def create_title(self, Var, type: str) -> str:
        """
        Create title of the plot

        Parameters
        ----------
        Var: MetricVar
            variable for a metric
        type: str
            type of plot
        """
        parts = [globals._metric_name[Var.metric]]
        parts.extend(QA4SMPlotter._get_parts_name(Var=Var, type=type))
        title = QA4SMPlotter._titles_lut(type=type).format(*parts)

        return title

    def create_label(self, Var) -> str:
        """
        Create y-label of the plot

        Parameters
        ----------

        Var: MetricVar
            variable for a metric

        Returns
        -------
        label: str
            y-label for the plot
        """
        parts = [globals._metric_name[Var.metric]]
        return "{}".format(*parts)

    def get_metric_vars(self, generic_metric: str) -> Dict[str, str]:
        _dict = {}

        _df = self.get_metric_df(generic_metric=generic_metric)
        for dataset in self.get_datasets_from_df(_df):
            for ds_combi, specific_metric in self.metrics_ds_grouped_lut(
            )[generic_metric].items():
                if dataset in ds_combi:
                    _Var = hdl.MetricVariable(varname=specific_metric,
                                              global_attrs=self.ds.attrs)
                    if _Var.values == None:
                        _Var.values = _df.loc[:, (dataset, slice(None))]

                    _dict[dataset] = _Var

        return _dict

    def get_legend_entries(self, generic_metric: str) -> Dict[str, str]:
        return {
            f'{Var.metric_ds[0]}-{Var.metric_ds[1]["short_name"]}':
            self.cbp.label_template.format(
                dataset_name=Var.metric_ds[1]["pretty_name"],
                unit=Var.get_varmeta()[3][1]["mu"] if Var.get_varmeta()[3] else Var.metric_ds[1]["mu"])
            for Var in self.get_metric_vars(generic_metric).values()
        }

    def _load_vars(self, empty=False, only_metrics=False) -> list:
        """
        Create a list of common variables and fill each with values

        Parameters
        ----------
        empty : bool, default is False
            if True, Var.values is an empty dataframe
        only_metrics : bool, default is False
            if True, only variables for metric scores are kept (i.e. not gpi, idx ...)

        Returns
        -------
        vars : list
            list of QA4SMVariable objects for the validation variables
        """
        vars = []
        for varname in self.metric_kinds_available:
            df = self.get_metric_df(generic_metric=varname)
            if empty:
                values = None
            else:
                # lat, lon are in varnames but not in datasframe (as they are the index)
                try:
                    # values = df
                    values = df
                except:  # KeyError:
                    values = None

            Var = hdl.QA4SMVariable(varname, self.ds.attrs,
                                    values=df).initialize()

            if only_metrics and isinstance(Var, hdl.MetricVariable):
                vars.append(Var)
            elif not only_metrics:
                vars.append(Var)

        return vars

    def _iter_vars(self,
                   type: str = None,
                   name: str = None,
                   filter_parms: dict = None) -> iter:
        """
        Iter through QA4SMVariable objects that are in the file

        Parameters
        ----------
        type : str, default is None
            One of 'metric', 'ci', 'metadata' can be specified to only iterate through the specific group
        name : str, default is None
            yield a specific variable by its name
        filter_parms : dict
            dictionary with QA4SMVariable attributes as keys and filter value as values (e.g. {g: 0})
        """
        type_lut = {
            "metric": hdl.MetricVariable,
            "ci": hdl.ConfidenceInterval,
            "metadata": hdl.Metadata,
        }
        for Var in self._load_vars():
            if name:
                if name in [Var.varname, Var.pretty_name]:
                    yield Var
                    break
                else:
                    continue
            if type and not isinstance(Var, type_lut[type]):
                continue
            if filter_parms:
                for key, val in filter_parms.items():
                    if getattr(Var,
                               key) == val:  # check all attribute individually
                        check = True
                    else:
                        check = False  # does not match requirements
                        break
                if check != True:
                    continue

            yield Var

    """def plot_cbp(
        self,
        chosen_metric: str,
        stability: bool,
        out_name: Optional[Union[List, List[str]]] = None
    ) -> matplotlib.figure.Figure:
        
        Plot a Clustered Boxplot for a chosen metric

        Parameters
        ----------
        chosen_metric : str
            name of the metric
        out_name : str or list of str, optional
            name of the output file. Default is None

        Returns
        -------
        fig : matplotlib.figure.Figure
            the boxplot

        
        anchor_list = None

        def get_metric_vars(
                generic_metric: str) -> Dict[str, hdl.MetricVariable]:
            _dict = {}

            for dataset in self.get_datasets_from_df(metric_df):
                for ds_combi, specific_metric in self.metrics_ds_grouped_lut(
                )[generic_metric].items():
                    if dataset in ds_combi:
                        _Var = hdl.MetricVariable(varname=specific_metric,
                                                  global_attrs=self.ds.attrs)
                        if _Var.values == None:
                            _Var.values = metric_df.loc[:,
                                                        (dataset, slice(None))]

                        _dict[(_Var.get_varmeta()[0][0], _Var.get_varmeta()[1][0])] = _Var

            return _dict

        def get_legend_entries(cbp_obj: ClusteredBoxPlot,
                               generic_metric: str) -> Dict[str, str]:
            unit = Var.metric_ds[1]["mu"]
            _, _, _ scl_ref, _ = Var.get_varmeta()
            return {
                f'{Var.metric_ds[0]} & {Var.metric_ds[1]["short_name"]}':
                cbp_obj.label_template.format(
                    dataset_name=Var.metric_ds[1]["pretty_name"],
                    unit=Var.metric_ds[1]["mu"])
                for Var in get_metric_vars(generic_metric).values()
            }
        
        def get_legend_entries_from_vars(cbp_obj: ClusteredBoxPlot, 
                                         vars: dict) -> Dict[str, str]:
            return {f'{i}':cbp_obj.label_template.format(
                    dataset_name=Var.metric_ds[1]["pretty_name"], #Check if scaled for units
                    unit=Var.metric_ds[1]["mu"] if not Var.get_varmeta()[3] else Var.get_varmeta()[3][1]["mu"]) if Var.metric_ds[1]['short_name'] in i 

                    else self.cbp.label_template.format(
                    dataset_name=Var.ref_ds[1]["pretty_name"],
                    unit=Var.ref_ds[1]["mu"] if not Var.get_varmeta()[3] else Var.get_varmeta()[3][1]["mu"]) if Var.ref_ds[1]['short_name'] in i 

                    else 'Dataset not in Data' for (i, Var) in vars.items()}

        def sanitize_dataframe(df: pd.DataFrame,
                               column_threshold: float = 0.1,
                               row_threshold_fraction: float = 0.8,
                               keep_empty_cols: bool = True) -> pd.DataFrame:
            
            Sanitizes a DataFrame by dropping columns and rows based on non-NaN thresholds.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to sanitize

            column_threshold : float, optional
                Fraction of non-NaN values in a column to keep it. Default is 0.1

            row_threshold_fraction : float, optional
                Fraction of non-NaN values in a row to keep it. Default is 0.8

            keep_empty_cols : bool
                Whether to keep column names that have non-NaNs below the threshold, but fill them with exclusively NaN. Default is True.
                This is done for the intra-annual metrics, where each month/season should be represented in the plot
                even if there is no data for a specific month/dataset in the end. The opposite is true for stability metrics.
                However if the sanitization leads to an empty dataframe the columns are kept to produce a plot.

            Returns
            -------
            df_sanitized : pd.DataFrame
                Sanitized DataFrame
            

            min_non_nan_columns = int(column_threshold * len(df))

            columns_to_keep = df.columns[df.notna().sum() >=
                                         min_non_nan_columns]
            columns_to_drop = df.columns[df.notna().sum() <
                                         min_non_nan_columns]

            df_sanitized = df[columns_to_keep]

            min_non_nan_rows = int(row_threshold_fraction *
                                   len(df_sanitized.columns))

            df_sanitized = df_sanitized.dropna(thresh=min_non_nan_rows)
            df_sanitized.dropna(inplace=True)

            # Return early if we don't need to keep empty columns and have data
            if not keep_empty_cols and not df_sanitized.empty:
                return df_sanitized

            for col in columns_to_drop:
                df_sanitized[col] = np.nan

            # Reorder the columns to match the original DataFrame
            df_sanitized = df_sanitized[df.columns]

            return df_sanitized

        metric_df = self.get_metric_df(chosen_metric)
        Vars = get_metric_vars(chosen_metric)
        
        anchor_list = None
        if stability:
            # get the first dataset to deduce the number of anchors - important for the boxplot setup
            unique_groups = metric_df.columns.get_level_values(0).unique()
            first_df = metric_df.loc[:,
                                     metric_df.columns.get_level_values(0) ==
                                     unique_groups[0]]
            first_df = sanitize_dataframe(first_df, keep_empty_cols=False)
            anchor_number = len(first_df.columns)
            anchor_list = np.arange(anchor_number).astype(float)

        # check also for empty anchor_list
        if anchor_list is None or (isinstance(anchor_list, np.ndarray) and anchor_list.size == 0):
            anchor_list = self.cbp.anchor_list

        figwidth = globals.boxplot_width_vertical * (len(metric_df.columns) + 1
                                            )  # otherwise it's too narrow
        figsize = [figwidth, globals.boxplot_height_vertical]
        fig_kwargs = {
            'figsize': figsize,
            'dpi': 'figure',
            'bbox_inches': 'tight'
        }

        legend_entries = get_legend_entries_from_vars(cbp_obj=self.cbp,
                                            vars=Vars)

        cbp_fig = self.cbp.figure_template(incl_median_iqr_n_axs=False,
                                           fig_kwargs=fig_kwargs)

        legend_handles = []
        df_whole = None
        for dc_num, (dc_val_name, Var) in enumerate(Vars.items()):
            _df = Var.values  # get the dataframe for the specific metric, potentially with NaNs
            _df.columns = pd.MultiIndex.from_tuples([(f"{Var.get_varmeta()[0][0]} & {Var.get_varmeta()[1][0]}", l2) for l0, l1, l2 in Var.values.columns],
                                                    names=["datasets", "time"])
            if not stability:
                _df = sanitize_dataframe(
                    _df, keep_empty_cols=True)  # sanitize the dataframe
            else:
                _df = sanitize_dataframe(
                    _df, keep_empty_cols=False)  # remove redundant columns

            df_long = _df.stack(level=0)
            df_long.index = df_long.index.set_names(["lat", "lon", "gpi", "dataset"])
            df_long = df_long.reset_index()
            df_long = df_long.melt(id_vars=["lat", "lon", "gpi", "dataset"], var_name="year", value_name="value")
            if isinstance(df_whole, pd.DataFrame):
                df_whole = pd.concat([df_whole, df_long], ignore_index=True)
                df_long=None
            else:
                df_whole = df_long
                df_long=None

        box = sns.boxplot(x="year",
                              y="value",
                              hue="dataset",
                              data=df_whole,
                              palette="Set2",
                              ax=cbp_fig.ax_box,
                              showfliers=False,
                              orient="v",
                              widths=0.8/df_whole.dataset.nunique(),
                              dodge=True)
            
        box.tick_params(labelsize=globals.fontsize_ticklabel)
        cbp_fig.ax_box.legend(loc=plm.best_legend_pos_exclude_list(cbp_fig.ax_box), fontsize=globals.fontsize_legend)

        cbp_fig.ax_box.legend(
            handles=legend_handles,
            loc=plm.best_legend_pos_exclude_list(cbp_fig.ax_box),
            fontsize=globals.fontsize_legend)
        
        QA4SMPlotter._append_legend_title(cbp_fig, cbp_fig.ax_box, Var)

        xtick_pos = self.cbp.centers_and_widths(anchor_list=anchor_list,
                                                no_of_ds=1,
                                                space_per_box_cluster=0.7,
                                                rel_indiv_box_width=0.8)
        cbp_fig.ax_box.set_xticks([])
        cbp_fig.ax_box.set_xticklabels([])
        cbp_fig.ax_box.set_xticks(xtick_pos[0].centers)

        def get_xtick_labels(df: pd.DataFrame) -> List:
            _count_dict = df.count().to_dict()
            return [
                f"{tsw[1]}\nEmpty" if count == 0 else f"{tsw[1]}"
                for tsw, count in _count_dict.items()
            ]

        xtick_labels = get_xtick_labels(_df)

        if len(xtick_labels) > 19 and stability:
            xtick_labels = [label.replace("\n", " ") for label in xtick_labels]
            cbp_fig.ax_box.set_xticklabels(xtick_labels)
            cbp_fig.ax_box.tick_params(axis='x', rotation=315,
                                       labelsize=globals.fontsize_ticklabel)
        else:
            cbp_fig.ax_box.set_xticklabels(xtick_labels)
        cbp_fig.ax_box.tick_params(
            axis='both',
            labelsize=globals.fontsize_ticklabel)

        _dummy_xticks = [
            cbp_fig.ax_box.axvline(x=(a + b) / 2, color='lightgrey') for a, b
            in zip(xtick_pos[0].centers[:-1], xtick_pos[0].centers[1:])
        ]

        title = self.create_title(Var, type='boxplot_basic')

        def get_valid_gpis(df: pd.DataFrame) -> int:

            try:
                out = list({x for x in df.count() if x > 0})[0]
            except IndexError:  # if all values are NaN
                out = 0
            return out

        title = title + f'\n for the same {get_valid_gpis(_df)} out of {len(metric_df)} GPIs\n'

        cbp_fig.fig.suptitle(
            title,
            fontsize=globals.fontsize_title)

        cbp_fig.ax_box.set_ylabel(
            self.create_label(Var),
            fontsize=globals.fontsize_label,
        )

        spth = [
            Path(
                f"{globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric = chosen_metric, filetype = 'png')}"
            )
        ]
        if out_name:
            spth = out_name

        [
            cbp_fig.fig.savefig(
                fname=outname,
                dpi=fig_kwargs['dpi'],
                bbox_inches=fig_kwargs['bbox_inches'],
            ) for outname in spth
        ]

        return cbp_fig.fig"""
    
    def plot_cbp(
        self,
        chosen_metric: str,
        stability: bool,
        out_name: Optional[Union[List, List[str]]] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot a Clustered Boxplot for a chosen metric

        Parameters
        ----------
        chosen_metric : str
            name of the metric
        out_name : str or list of str, optional
            name of the output file. Default is None

        Returns
        -------
        fig : matplotlib.figure.Figure
            the boxplot

        """
        anchor_list = None

        def get_metric_vars(
                generic_metric: str) -> Dict[str, hdl.MetricVariable]:
            _dict = {}

            for dataset in self.get_datasets_from_df(metric_df):
                for ds_combi, specific_metric in self.metrics_ds_grouped_lut(
                )[generic_metric].items():
                    if dataset in ds_combi:
                        _Var = hdl.MetricVariable(varname=specific_metric,
                                                  global_attrs=self.ds.attrs)
                        if _Var.values == None:
                            _Var.values = metric_df.loc[:,
                                                        (dataset, slice(None))]

                        _dict[(_Var.get_varmeta()[0][0], _Var.get_varmeta()[1][0])] = _Var

            return _dict

        def sanitize_dataframe(df: pd.DataFrame,
                               column_threshold: float = 0.1,
                               row_threshold_fraction: float = 0.8,
                               keep_empty_cols: bool = True) -> pd.DataFrame:
            """
            Sanitizes a DataFrame by dropping columns and rows based on non-NaN thresholds.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to sanitize

            column_threshold : float, optional
                Fraction of non-NaN values in a column to keep it. Default is 0.1

            row_threshold_fraction : float, optional
                Fraction of non-NaN values in a row to keep it. Default is 0.8

            keep_empty_cols : bool
                Whether to keep column names that have non-NaNs below the threshold, but fill them with exclusively NaN. Default is True.
                This is done for the intra-annual metrics, where each month/season should be represented in the plot
                even if there is no data for a specific month/dataset in the end. The opposite is true for stability metrics.
                However if the sanitization leads to an empty dataframe the columns are kept to produce a plot.

            Returns
            -------
            df_sanitized : pd.DataFrame
                Sanitized DataFrame
            """

            min_non_nan_columns = int(column_threshold * len(df))

            columns_to_keep = df.columns[df.notna().sum() >=
                                         min_non_nan_columns]
            columns_to_drop = df.columns[df.notna().sum() <
                                         min_non_nan_columns]

            df_sanitized = df[columns_to_keep]

            min_non_nan_rows = int(row_threshold_fraction *
                                   len(df_sanitized.columns))

            df_sanitized = df_sanitized.dropna(thresh=min_non_nan_rows)
            df_sanitized.dropna(inplace=True)

            # Return early if we don't need to keep empty columns and have data
            if not keep_empty_cols and not df_sanitized.empty:
                return df_sanitized

            for col in columns_to_drop:
                df_sanitized[col] = np.nan

            # Reorder the columns to match the original DataFrame
            df_sanitized = df_sanitized[df.columns]

            return df_sanitized

        metric_df = self.get_metric_df(chosen_metric)
        Vars = get_metric_vars(chosen_metric)
        
        anchor_list = None
        if stability:
            # get the first dataset to deduce the number of anchors - important for the boxplot setup
            unique_groups = metric_df.columns.get_level_values(0).unique()
            first_df = metric_df.loc[:,
                                     metric_df.columns.get_level_values(0) ==
                                     unique_groups[0]]
            first_df = sanitize_dataframe(first_df, keep_empty_cols=False)
            anchor_number = len(first_df.columns)
            anchor_list = np.arange(anchor_number).astype(float)

        # check also for empty anchor_list
        if anchor_list is None or (isinstance(anchor_list, np.ndarray) and anchor_list.size == 0):
            anchor_list = self.cbp.anchor_list

        cbp_fig = self.cbp.figure_template(incl_median_iqr_n_axs=False)

        metric_df = self.get_metric_df(chosen_metric)
        metric_df.columns = pd.MultiIndex.from_tuples([(f"{l0[:1]} & {l1[:1]}", l2) for l0, l1, l2 in metric_df.columns],
                                                    names=["datasets", "time"])
        df_long = metric_df.stack(level=0, future_stack=True)
        df_long.index = df_long.index.set_names(["lat", "lon", "gpi", "dataset"])
        df_long = df_long.reset_index()
        df_long = df_long.melt(id_vars=["lat", "lon", "gpi", "dataset"], var_name="year", value_name="value").sort_values("dataset", ascending=True)
        
        unique_combos = df_long["dataset"].unique()
        palette = get_palette_for(unique_combos)

        n_lines = len(cbp_fig.ax_box.lines)
        box = sns.boxplot(x="year",
                              y="value",
                              hue="dataset",
                              data=df_long,
                              palette=palette,
                              ax=cbp_fig.ax_box,
                              showfliers=False,
                              orient="v",
                              widths=0.8/df_long.dataset.nunique(),
                              dodge=True)
        plm.capsizing(box, n_lines)
            
        box.tick_params(labelsize=globals.fontsize_ticklabel)
        cbp_fig.ax_box.legend(loc=plm.best_legend_pos_exclude_list(cbp_fig.ax_box), fontsize=globals.fontsize_legend)
        
        Var = hdl.MetricVariable(varname=self.metrics_ds_grouped_lut()[chosen_metric][list(self.metrics_ds_grouped_lut()[chosen_metric].keys())[0]], global_attrs=self.ds.attrs)
        QA4SMPlotter._append_legend_title(cbp_fig, cbp_fig.ax_box, Var)

        # Styling of axis
        cbp_fig.ax_box.set_ylabel(self.create_label(Var), fontsize=globals.fontsize_label)
        cbp_fig.ax_box.xaxis.label.set_fontsize(globals.fontsize_label)

        ticks = cbp_fig.ax_box.get_xticks()
        midpoints = [(ticks[i] + ticks[i + 1]) / 2 for i in range(len(ticks) - 1)]
        cbp_fig.ax_box.xaxis.set_minor_locator(plt.FixedLocator(midpoints))
        cbp_fig.ax_box.grid(which='minor', color='gray', linestyle='dotted', linewidth=0.8)
        cbp_fig.ax_box.grid(which='major', axis='y', color='gray', linestyle='-', linewidth=0.8)
        cbp_fig.ax_box.grid(which='major', axis='x', visible=False)

        # Increase fig dimensions for large amount of periods
        n_periods = len(ticks)
        if n_periods > globals.period_bin_th: 
            dims = [globals.boxplot_width_vertical*n_periods/globals.period_bin_th,globals.boxplot_height_vertical]
        else:
            dims = [globals.boxplot_width_vertical,globals.boxplot_height_vertical]
        cbp_fig.fig.set_figwidth(dims[0])
        cbp_fig.fig.set_figheight(dims[1])

        if len(ticks)>globals.no_growth_th_v:
            cbp_fig.ax_box.set_xlim(ticks[0]-(ticks[1]-ticks[0])/2, ticks[-1]+(ticks[-1]-ticks[-2])/2)
        else:
            cbp_fig.ax_box.set_xlim((ticks[0]+ticks[-1]-globals.no_growth_th_v-1)/2, (ticks[0]+ticks[-1]+globals.no_growth_th_v+1)/2)

        title = self.create_title(Var, type='boxplot_basic')

        def get_valid_gpis(df: pd.DataFrame) -> int:

            try:
                out = list({x for x in df.count() if x > 0})[0]
            except IndexError:  # if all values are NaN
                out = 0
            return out

        title = title + f' for the same {get_valid_gpis(metric_df)} out of {len(metric_df)} GPIs'

        QA4SMPlotter._set_wrapped_title(cbp_fig.fig, cbp_fig.ax_box, title)

        if globals.draw_logo:
            plm.add_logo_to_figure(
                fig=cbp_fig.fig,
                logo_path=globals.logo_pth,
                position=globals.logo_position,
                offset=globals.logo_offset_comp_plots,
                size=globals.logo_size,
            )

        spth = [
            Path(
                f"{globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric = chosen_metric, filetype = 'png')}"
            )
        ]
        if out_name:
            spth = out_name

        [
            cbp_fig.fig.savefig(
                fname=outname,
                dpi="figure", 
                bbox_inches='tight',
            ) for outname in spth
        ]

        return cbp_fig.fig

