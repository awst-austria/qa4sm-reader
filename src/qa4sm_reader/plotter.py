# -*- coding: utf-8 -*-
from pathlib import Path
import seaborn as sns
import pandas as pd

from qa4sm_reader.img import QA4SMImg
import qa4sm_reader.globals as globals
from qa4sm_reader.plot_utils import *

from warnings import warn


class QA4SMPlotter():
    """
    Class to create image files of plots from the validation results in a QA4SMImage
    """
    def __init__(self, image, out_dir:str=None):
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
        except:
            warn("The initialized QA4SMImg object has not been loaded. 'load_data' needs to be "
                 "set to 'True' in the initialization of the Image.")

    def get_dir(self, out_dir:str) -> Path:
        """Use output path if specified, otherwise same directory as the one storing the netCDF file"""
        if out_dir:
            out_dir = Path(out_dir)  # use directory if specified
            if not out_dir.exists():
                out_dir.mkdir()  # make if not existing
        else:
            out_dir = self.img.filepath.parent  # use default otherwise

        return out_dir

    def _standard_filename(self, out_name:str, out_type:str='png') -> Path:
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
    def _box_stats(ds:pd.Series, med:bool=True, iqr:bool=True, count:bool=True) -> str:
        """
        Create the metric part with stats of the box (axis) caption

        Parameters
        ----------
        ds: pd.Series
            data on which stats are found
        med: bool
        iqr: bool
        count: bool
            statistics

        Returns
        -------
        stats: str
            caption with summary stats
        """
        # interquartile range
        iqr = ds.quantile(q=[0.75,0.25]).diff()
        iqr = abs(float(iqr.loc[0.25]))

        met_str = []
        if med:
            met_str.append('median: {:.3g}'.format(ds.median()))
        if iqr:
            met_str.append('IQR: {:.3g}'.format(iqr))
        if count:
            met_str.append('N: {:d}'.format(ds.count()))
        stats = '\n'.join(met_str)

        return stats

    @staticmethod
    def _box_caption(Var, tc:bool=False) -> str:
        """
        Create the dataset part of the box (axis) caption

        Parameters
        ----------
        Var: MetricVar
            variable for a metric
        tc: bool, default is False
            True if TC. Then, caption starts with "Other Data:"

        Returns
        -------
        capt: str
            box caption
        """
        ref_meta, mds_meta, other_meta = Var.get_varmeta()
        ds_parts = []
        id, meta = mds_meta
        if tc:
            id, meta = other_meta
        ds_parts.append('{}-{}\n({})'.format(
            id, meta['pretty_name'], meta['pretty_version']))
        capt = '\n and \n'.join(ds_parts)

        if tc:
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
        parts: list
            list of parts for title
        """
        parts = []
        ref, mds, other = [meta for meta in Var.get_varmeta()]
        if type == 'boxplot_basic':
            parts.append(ref[0])
            parts.extend([ref[1]['pretty_name'], ref[1]['pretty_version']])

        elif type in ['boxplot_tc', 'mapplot_basic', 'mapplot_tc']:
            parts.append(mds[0])
            parts.extend([mds[1]['pretty_name'], mds[1]['pretty_version']])
            parts.append(ref[0])
            parts.extend([ref[1]['pretty_name'], ref[1]['pretty_version']])

            if type == 'mapplot_tc':
                parts.append(other[0])
                parts.extend([other[1]['pretty_name'], other[1]['pretty_version']])

        return parts

    @staticmethod
    def _titles_lut(type:str) -> str:
        """
        Lookup table for plot titles

        Parameters
        ----------
        type: str
            type of plot
        """
        titles = {'boxplot_basic': 'Intercomparison of \n{} \nwith {}-{} ({}) \nas the reference',
                  'boxplot_tc': 'Intercomparison of \n{} \nfor {}-{} ({}) \nwith {}-{} ({}) \nas the reference',
                  'mapplot_basic': '{} for {}-{} ({}) with {}-{} ({}) as the reference',
                  'mapplot_tc': '{} for {}-{} ({}) with {}-{} ({}) and {}-{} ({}) as the references'}

        try:
            return titles[type]

        except KeyError as e:
            message = "type '{}' is not in the lookup table".format(type)
            warn(message)

    @staticmethod  # todo: cange file names and convention in qa4sm
    def _filenames_lut(type:str) -> str:
        """
        Lookup table for file names

        Parameters
        ----------
        type: str
            type of plot
        """
        # we stick to old naming convention
        names = {'boxplot_basic': 'boxplot_{}',
                 'mapplot_common': 'overview_{}',
                 'boxplot_tc': 'boxplot_{}_for_{}-{}',
                 'mapplot_double': 'overview_{}-{}_and_{}-{}_{}',
                 'mapplot_tc': 'overview_{}-{}_and_{}-{}_and_{}-{}_{}_for_{}-{}'}

        try:
            return names[type]

        except KeyError as e:
            message = "type '{}' is not in the lookup table".format(type)
            warn(message)

    def create_title(self, Var, type:str) -> str:
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

        return title

    def create_filename(self, Var, type:str) -> str:
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
        ref_meta, mds_meta, other_meta = Var.get_varmeta()
        # fetch parts of the name for the variable
        if not type in ["mapplot_tc", "mapplot_double"]:
            parts = [Var.metric]
            if mds_meta:
                parts.extend([mds_meta[0], mds_meta[1]['short_name']])
        else:
            parts = [ref_meta[0], ref_meta[1]['short_name']]
            if type == "mapplot_tc":
                # necessary to respect old naming convention
                for dss in  Var.other_dss:
                    parts.extend([dss[0], dss[1]['short_name']])
                parts.extend([Var.metric, mds_meta[0], mds_meta[1]['short_name']])
            parts.extend([mds_meta[0], mds_meta[1]['short_name'], Var.metric])

        name = name.format(*parts)

        return name

    def _yield_values(self, metric:str, tc:bool=False) -> tuple:
        """
        Get iterable with pandas dataframes for all variables of a metric to plot

        Parameters
        ----------
        metric: str
            metric name
        add_stats : bool, optional (default: from globals)
            Add stats of median, iqr and N to the box bottom.
        tc: bool, default is False
            True if TC. Then, caption starts with "Other Data:"

        Yield
        -----
        df: pd.DataFrame
            dataframe with variable values and caption name
        Var: QA4SMMetricVariable
            variable corresponding to the dataframe
        """
        Vars = self.img._iter_vars(**{'metric':metric})

        for n, Var in enumerate(Vars):
            values = Var.values[Var.varname]

            # changes if it's a common-type Var
            if Var.g == 0:
                box_cap_ds = 'All datasets'
            else:
                box_cap_ds = self._box_caption(Var, tc=tc)
            # setting in global for caption stats
            if globals.boxplot_printnumbers:
                box_stats = self._box_stats(values)
                box_cap = '{}\n{}'.format(box_cap_ds, box_stats)
            else:
                box_cap = box_cap_ds
            df = values.to_frame(box_cap)

            yield df, Var


    def _boxplot_definition(self, metric:str,
                            df:pd.DataFrame,
                            type:str,
                            offset=0.08,
                            Var=None,
                            **kwargs) -> tuple:
        """
        Define parameters of plot

        Parameters
        ----------
        df: pd.DataFrame
            dataframe to plot
        type: str
            one of _titles_lut
        offset: float
            offset of boxplots
        Var: QA4SMMetricVariable, optional. Default is None
            Specified in case mds meta is needed
        """
        # plot label
        parts = [globals._metric_name[metric]]
        parts.append(globals._metric_description[metric].format(
            globals._metric_units[self.ref['short_name']]))
        label = "{}{}".format(*parts)
        # generate plot
        figwidth = globals.boxplot_width * (len(df.columns) + 1)
        figsize = [figwidth, globals.boxplot_height]
        fig, ax = boxplot(df=df, label=label, figsize=figsize, dpi=globals.dpi)
        if not Var:
            # when we only need reference dataset from variables (i.e. is the same):
            for Var in self.img._iter_vars(**{'metric':metric}):
                Var = Var
                break
        title = self.create_title(Var, type=type)
        ax.set_title(title, pad=globals.title_pad)
        # add watermark
        if Var.g == 0:
            offset = 0.03  # offset smaller as common metrics have a shorter caption
        if globals.watermark_pos not in [None, False]:
            make_watermark(fig, offset=offset)

        return fig, ax

    def _save_plot(self, out_name:str, out_types:str='png') -> list:
        """
        Save plot with name to self.out_dir

        Parameters
        ----------
        out_name: str
            name of output file
        out_types: str or list
            extensions which the files should be saved in

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        fnames = []
        if isinstance(out_types, str):
            out_types = [out_types]
        for ext in out_types:
            fname = self._standard_filename(out_name, out_type=ext)
            if fname.exists():
                warnings.warn('Overwriting file {}'.format(fname.name))
            plt.savefig(fname, dpi='figure', bbox_inches='tight')
            fnames.append(fname.absolute())

        return fnames

    def boxplot_basic(self, metric:str,
                      out_name:str=None,
                      out_types:str='png',
                      save_files:bool=False,
                      **plotting_kwargs) -> list:
        """
        Creates a boxplot for common and double metrics. Saves a figure and returns Matplotlib fig and ax objects for
        further processing.

        Parameters
        ----------
        metric : str
            metric that is collected from the file for all datasets and combined
            into one plot.
        out_name: str
            name of output file
        out_types: str or list
            extensions which the files should be saved in
        save_file: bool, optional. Default is False
            wether to save the file in the output directory
        plotting_kwargs: arguments for _boxplot_definition function

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        fnames, values = [], []
        # we take the last iterated value for Var and use it for the file name
        for df, Var in self._yield_values(metric=metric):
            values.append(df)

        values = pd.concat(values)
        # values are all Nan or NaNf - not plotted
        if df.isnull().values.all():
            return None
        # create plot
        fig, ax = self._boxplot_definition(metric=metric,
                                           df=values,
                                           type='boxplot_basic',
                                           **plotting_kwargs)
        if not out_name:
            out_name = self.create_filename(Var, type='boxplot_basic')
        # save or return plotting objects
        if save_files:
            fnames = self._save_plot(out_name, out_types=out_types)
            plt.close('all')

            return fnames

        else:
            return fig, ax

    def boxplot_tc(self, metric:str,
                   out_name:str=None,
                   out_types:str='png',
                   save_files:bool=False,
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
        out_types: str or list
            extensions which the files should be saved in
        save_file: bool, optional. Default is False
            wether to save the file in the output directory
        plotting_kwargs: arguments for _boxplot_definition function

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        fnames = []
        metric_tc = {}  # group Vars relative to the same dataset
        for df, Var in self._yield_values(metric=metric, tc=True):
            ref_meta, mds_meta, other_meta = Var.get_varmeta()
            id, names = mds_meta
            if id in metric_tc.keys():
                metric_tc[id][0].append(df)
            else:
                metric_tc[id] = [df], Var

        for dfs, Var in metric_tc.values():
            df = pd.concat(dfs)
            # values are all Nan or NaNf - not plotted
            if df.isnull().values.all():
                continue
            # create plot
            fig, ax = self._boxplot_definition(metric=metric,
                                               df=df,
                                               type='boxplot_tc',
                                               Var=Var,
                                               **plotting_kwargs)
            # save. Below workaround to avoid same names
            if not out_name:
                save_name = self.create_filename(Var, type='boxplot_tc')
            else:
                save_name = out_name
            # save or return plotting objects
            if save_files:
                fns = self._save_plot(save_name, out_types=out_types)
                fnames.extend(fns)
                plt.close('all')

        if save_files:
            return fnames

    def mapplot_var(self, Var,
                    out_name:str=None,
                    out_types:str='png',
                    save_files:bool=False,
                    **plotting_kwargs) -> list:
        """
        Plots values to a map, using the values as color. Plots a scatterplot for
        ISMN and a image plot for other input values.

        Parameters
        ----------
        var : QA4SMMetricVariab;e
            Var in the image to make the map for.
        out_name: str
            name of output file
        out_types: str or list
            extensions which the files should be saved in
        save_file: bool, optional. Default is False
            wether to save the file in the output directory
        plotting_kwargs: arguments for mapplot function

        Returns
        -------
        fnames: list
            list of file names with all the extensions
        """
        ref_meta, mds_meta, other_meta = Var.get_varmeta()
        metric = Var.metric
        ref_grid_stepsize = self.img.ref_dataset_grid_stepsize

        # create mapplot
        fig, ax = mapplot(df=Var.values[Var.varname],
                          metric=metric,
                          ref_short=ref_meta[1]['short_name'],
                          ref_grid_stepsize=ref_grid_stepsize,
                          plot_extent=None,  # if None, extent is sutomatically adjusted (as opposed to img.extent)
                          **plotting_kwargs)

        # title and plot settings depend on the metric group
        if Var.g == 0:
            title = "{} between all datasets".format(globals._metric_name[metric])
            out_name = self.create_filename(Var, type='mapplot_common')
        elif Var.g == 2:
            title = self.create_title(Var=Var, type='mapplot_basic')
            out_name = self.create_filename(Var, type='mapplot_double')
        else:
            title = self.create_title(Var=Var, type='mapplot_tc')
            out_name = self.create_filename(Var, type='mapplot_tc')

        # use title for plot, make watermark
        ax.set_title(title, pad=globals.title_pad)
        if globals.watermark_pos not in [None, False]:
            make_watermark(fig, globals.watermark_pos, for_map=True)

        # save file or just return the image
        if save_files:
            fnames = self._save_plot(out_name, out_types=out_types)

            return fnames

        else:
            return fig, ax

    def mapplot_metric(self, metric:str,
                       out_types:str='png',
                       save_files:bool=False,
                       **plotting_kwargs) -> list:
        """
        Mapplot for all variables for a given metric in the loaded file.

        Parameters
        ----------
        metric : str
            Name of a metric. File is searched for variables for that metric.
        out_name: str
            name of output file
        out_types: str or list
            extensions which the files should be saved in
        save_file: bool, optional. Default is False
            wether to save the file in the output directory
        plotting_kwargs: arguments for mapplot function

        Returns
        -------
        fnames : list
            List of files that were created
        """
        fnames = []
        for Var in self.img._iter_vars(**{'metric':metric}):
            if not Var.values.isnull().values.all():
                fns = self.mapplot_var(Var,
                                       out_name=None,
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

    def plot_metric(self, metric:str, out_types:str='png', save_all:bool=True, **plotting_kwargs) -> tuple:
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
        Metric = self.img.metrics[metric]
        if Metric.g == 0 or Metric.g == 2:
            fnames_bplot = self.boxplot_basic(metric=metric,
                                              out_types=out_types,
                                              save_files=save_all,
                                              **plotting_kwargs)
        elif Metric.g == 3:
            fnames_bplot = self.boxplot_tc(metric=metric,
                                           out_types=out_types,
                                           save_files=save_all,
                                           **plotting_kwargs)
        fnames_mapplot = self.mapplot_metric(metric=metric,
                                             out_types=out_types,
                                             save_files=save_all,
                                             **plotting_kwargs)

        return fnames_bplot, fnames_mapplot
