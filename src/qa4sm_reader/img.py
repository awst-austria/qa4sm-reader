# -*- coding: utf-8 -*-

from qa4sm_reader import globals
from qa4sm_reader.handlers import QA4SMDatasets, QA4SMMetricVariable, QA4SMMetric
from qa4sm_reader.plot_utils import _format_floats

from parse import *
from pathlib import Path
import numpy as np
import xarray as xr
from collections import OrderedDict
import itertools
import pandas as pd


def extract_periods(filepath) -> np.array:
    """Get periods from .nc"""
    dataset = xr.open_dataset(filepath)
    if globals.period_name in dataset.dims:
        return dataset[globals.period_name].values

    else:
        return np.array([None])


class QA4SMImg(object):
    """A tool to analyze the results of a validation, which are stored in a netCDF file."""
    def __init__(self, filepath,
                 period=None,
                 extent=None,
                 ignore_empty=True,
                 metrics=None,
                 index_names=globals.index_names,
                 load_data=True):
        """
        Initialise a common QA4SM results image.

        Parameters
        ----------
        filepath : str
            Path to the results netcdf file (as created by QA4SM)
        period : Any, optional (default: None)
            If results for multiple validation periods are stored in file,
            load this period.
        extent : tuple, optional (default: None)
            Area to subset the values for -> (min_lon, max_lon, min_lat, max_lat)
        ignore_empty : bool, optional (default: True)
            Ignore empty variables in the file.
        metrics : list or None, optional (default: None)
            Subset of the metrics to load from file, if None are passed, all
            are loaded.
        index_names : list, optional (default: ['lat', 'lon'] - as in globals.py)
            Names of dimension variables in x and y direction (lat, lon).
        load_data: bool, default is True
            if true, initialize all the datasets, variables and metadata
        """
        self.filepath = Path(filepath)
        self.index_names = index_names

        self.ignore_empty = ignore_empty
        self.ds = self._open_ds(extent=extent, period=period)
        self.extent = self._get_extent(extent=extent)  # get extent from .nc file if not specified
        self.datasets = QA4SMDatasets(self.ds.attrs)
        self.name = self.create_image_name()

        if load_data:
            self.varnames = list(self.ds.variables.keys())
            self.df = self._ds2df()
            self.vars = self._load_vars()
            self.metrics = self._load_metrics()
            self.common, self.double, self.triple = self.group_metrics(metrics)
            # this try here is to obey tests, withouth a necessity of changing and commiting test files again
            try:
                self.ref_dataset_grid_stepsize = self.ds.val_dc_dataset0_grid_stepsize
            except:
                self.ref_dataset_grid_stepsize = 'nan'

    def _open_ds(self, extent=None, period=None):
        """Open .nc as xarray datset, with selected extent"""
        dataset = xr.open_dataset(self.filepath)
        if period is not None:
            ds = dataset.sel(dict(period=period))
        else:
            ds = dataset
        # drop non-spatial variables (e.g.'time')
        if globals.time_name in ds.variables:
            ds = ds.drop_vars(globals.time_name)
        # geographical subset of the results
        if extent:
            lat, lon = globals.index_names
            mask = (ds[lon] >= extent[0]) & (ds[lon] <= extent[1]) &\
                   (ds[lat] >= extent[2]) & (ds[lat] <= extent[3])

            assert True in mask, "The selected subset is not overlapping the validation domain"

            return ds.where(mask, drop=True)

        else:
            return ds

    @property
    def has_CIs(self):
        """True if the validation result contains confidence intervals"""
        cis = False
        # check if there is any CI Var
        for Var in self._iter_vars():
            if Var.is_CI:
                cis = True
        return cis

    def create_image_name(self) -> str:
        """Create a unique name for the QA4SMImage from the netCDF file"""
        ref = self.datasets.ref['pretty_title']
        others = [other['pretty_title'] for other in self.datasets.others]

        name = "ref: {} v datasets: ".format(ref) + \
               ", ".join(others)

        return name

    def _get_extent(self, extent) -> tuple:
        """Get extent of the results from the netCDF file"""
        if not extent:
            lat, lon = globals.index_names
            lat_coord, lon_coord = self.ds[lat].values, self.ds[lon].values
            lons = min(lon_coord), max(lon_coord)
            lats = min(lat_coord), max(lat_coord)
            extent = lons + lats

        return extent

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
            list of QA4SMMetricVariable objects for the validation variables
        """
        vars = []
        for varname in self.varnames:
            if empty:
                values = None
            else:
                # lat, lon are in varnames but not in datasframe (as they are the index)
                try:
                    values = self.df[[varname]]
                except KeyError:
                    values = None

            try:
                Var = QA4SMMetricVariable(varname, self.ds.attrs, values=values)
                if self.ignore_empty and Var.isempty:  # check whether there are values
                    continue
            except IOError:
                Var = None
                continue

            if not Var is None:
                if only_metrics and Var.ismetric:
                    vars.append(Var)
                elif not only_metrics:
                    vars.append(Var)

        return vars

    def _load_metrics(self) -> dict:
        """
        Create a list of metrics for the file

        Returns
        -------
        Metrics : dict
            dictionary with shape {metric name: QA4SMMetric}
        """
        Metrics = {}
        all_groups = globals.metric_groups.values()
        for group in all_groups:
            for metric in group:
                metric_vars = []
                for Var in self._iter_vars(**{'metric': metric}):
                    metric_vars.append(Var)

                if metric_vars != []:
                    Metric = QA4SMMetric(metric, metric_vars)
                    Metrics[metric] = Metric

        return Metrics

    def _iter_vars(self, only_metrics=False, **filter_parms) -> iter:
        """
        Iter through QA4SMMetricVariable objects that are in the file

        Parameters
        ----------
        only_metrics: bool, optional. Default is Fales.
            If True, only Vars that belong to a group are taken
        **filter_parms : kwargs, dict
            dictionary with QA4SMMetricVariable attributes as keys and filter value as values (e.g. {g: 0})
        """
        for Var in self.vars:
            if only_metrics:
                if Var.g is None:
                    continue
            if filter_parms:
                for key, val in filter_parms.items():
                    if getattr(Var, key) == val:  # check all attribute individually
                        check = True
                    else:
                        check = False  # does not match requirements
                        break
                if check == True:
                    yield Var
            else:
                yield Var

    def _iter_metrics(self, **filter_parms) -> iter:
        """
        Iter through QA4SMMetric objects that are in the file

        Parameters
        ----------
        **filter_parms : kwargs, dict
            dictionary with QA4SMMetric attributes as keys and filter value as values (e.g. {g: 0})
        """
        for Metric in self.metrics.values():
            for key, val in filter_parms.items():
                if val is None or getattr(Metric, key) == val:
                    yield Metric

    def group_vars(self, **filter_parms):
        """
        Return a list of QA4SMMetricVariable that match filters

        Parameters
        ----------
        **filter_parms : kwargs, dict
            dictionary with QA4SMMetricVariable attributes as keys and filter value as values (e.g. {g: 0})
        """
        vars = []
        for Var in self._iter_vars(**filter_parms):
            vars.append(Var)

        return vars

    def group_metrics(self, metrics:list=None) -> (dict, dict, dict):
        """
        Load and group all metrics from file

        Parameters
        ----------
        metrics: list or None
            if list, only metrics in the list are grouped
        """
        common, double, triple = {},{},{}

        # fetch Metrics
        if metrics is None:
            metrics = self.metrics.keys()

        # fill dictionaries
        for metric in metrics:
            Metric = self.metrics[metric]
            if Metric.g == 0:
                common[metric] = Metric
            elif Metric.g == 2:
                double[metric] = Metric
            elif Metric.g == 3:
                triple[metric] = Metric

        return common, double, triple

    def _ds2df(self, varnames:list=None) -> pd.DataFrame:
        """
        Return one or more or all variables in a single DataFrame.

        Parameters
        ----------
        varnames : list or None
            list of QA4SMMetricVariables to be placed in the DataFrame

        Return
        ------
        df : pd.DataFrame
            DataFrame with Var name as column names
        """
        try:
            if varnames is None:
                if globals.time_name in self.varnames:
                    if self.ds[globals.time_name].values.size == 0:
                         self.ds = self.ds.drop_vars(globals.time_name)
                df = self.ds.to_dataframe()
            else:
                df = self.ds[self.index_names + varnames].to_dataframe()
                df.dropna(axis='index', subset=varnames, inplace=True)
        except KeyError as e:
            raise Exception("The variable name '{}' does not match any name in the input values.".format(e.args[0]))

        if isinstance(df.index, pd.MultiIndex):
            lat, lon = globals.index_names
            df[lat] = df.index.get_level_values(lat)
            df[lon] = df.index.get_level_values(lon)

        df.reset_index(drop=True, inplace=True)
        df = df.set_index(self.index_names)

        return df

    def metric_df(self, metrics:str or list):
        """
        Group all variables for the metric in a common data frame

        Parameters
        ---------
        metrics : str or list
            name(s) of the metrics to have in the DataFrame

        Returns
        -------
        df : pd.DataFrame
            A dataframe that contains all variables that describe the metric
            in the column
        """
        if isinstance(metrics, list):
            Vars = []
            for metric in metrics:
                Vars.extend(self.group_vars(**{'metric':metric}))
        else:
            Vars = self.group_vars(**{'metric':metrics})

        varnames = [Var.varname for Var in Vars]
        metrics_df = self._ds2df(varnames=varnames)

        return metrics_df

    def _metric_stats(self, metric, id=None)  -> list:
        """
        Provide a list with the metric summary statistics for each variable or for all variables
        where the dataset with id=id is the metric dataset.

        Parameters
        ----------
        metric : str
            A metric that is in the file (e.g. n_obs, R, ...)
        id: int
            dataset id

        Returns
        -------
        metric_stats : list
            List of (variable) lists with summary statistics
        """
        metric_stats = []
        if id:
            filters = {'metric':metric, 'id':id}
        else:
            filters = {'metric':metric}
        # get stats by metric
        for Var in self._iter_vars(**filters):
            # get interquartile range 
            values = Var.values[Var.varname]
            # take out variables with all NaN or NaNf
            if values.isnull().values.all():
                continue
            iqr = values.quantile(q=[0.75,0.25]).diff()
            iqr = abs(float(iqr.loc[0.25]))
            # find the statistics for the metric variable
            var_stats = [i for i in (values.mean(), values.median(), iqr)]
            if Var.g == 0:
                var_stats.append('All datasets')
                var_stats.extend([globals._metric_name[metric], Var.g])

            else:
                i, ds_name = Var.metric_ds
                if Var.g == 2:
                    var_stats.append('{}-{} ({})'.format(i, ds_name['short_name'], ds_name['pretty_version']))

                elif Var.g == 3:
                    o, other_ds = Var.other_ds
                    var_stats.append('{}-{} ({}); other ref: {}-{} ({})'.format(i, ds_name['short_name'],
                                                                                ds_name['pretty_version'],
                                                                                o, other_ds['short_name'],
                                                                                other_ds['pretty_version']))

                var_stats.extend([globals._metric_name[metric] + globals._metric_description_HTML[metric].format(
                    globals._metric_units_HTML[ds_name['short_name']]), Var.g])
            # put the separate variable statistics in the same list
            metric_stats.append(var_stats)

        return metric_stats
    
    def stats_df(self) -> pd.DataFrame:
        """
        Create a DataFrame with summary statistics for all the metrics

        Returns
        -------
        stats_df : pd.DataFrame
            Quick inspection table of the results.
        """
        stats = []
        # find stats for all the metrics
        for metric in self.metrics.keys():
            stats.extend(self._metric_stats(metric))
        # create a dataframe
        stats_df = pd.DataFrame(stats, columns = ['Mean', 'Median', 'IQ range', 'Dataset', 'Metric', 'Group'])
        stats_df.set_index('Metric', inplace=True)
        stats_df.sort_values(by='Group', inplace=True)
        # format the numbers for display
        stats_df = stats_df.applymap(_format_floats)

        return stats_df
