# -*- coding: utf-8 -*-

from qa4sm_reader.img import QA4SMImg
import os
import numpy as np
import pandas as pd
import unittest
from qa4sm_reader import globals


class TestQA4SMImgBasicIntercomp(unittest.TestCase):

    def setUp(self) -> None:
        self.testfile = '3-ERA5_LAND.swvl1_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'basic', self.testfile)
        self.img = QA4SMImg(self.testfile_path, ignore_empty=False)

    def test_load_data(self):
        unloaded = QA4SMImg(self.testfile_path, load_data=False)
        assert 'varnames' not in unloaded.__dict__.keys()

    def test_extent(self):
        extent = QA4SMImg(self.testfile_path, extent=(113.7, 123.7, -19.8, -9.8))
        assert self.img.extent != extent.extent
        assert self.img.extent == (113.7, 153.5, -43.1, -9.8)

    def test_metrics(self):
        metrics = QA4SMImg(self.testfile_path, metrics=['R'])
        assert metrics.common != self.img.common
        assert metrics.double != self.img.double
        assert 'R' in metrics.double.keys()

    def test_load_vars(self):
        Vars = self.img._load_vars()
        assert len(Vars) == len(self.img.varnames)
        Metr_Vars = self.img._load_vars(only_metrics=True)
        assert len(Metr_Vars) == len(Vars) - 3

    def test_iter_vars(self):
        for Var in self.img._iter_vars(only_metrics=True):
            assert Var.g in [0,2,3]
        for Var in self.img._iter_vars(**{'metric':'R'}):
            Var.varname in ['R_between_3-ERA5_LAND_and_2-SMOS', 'R_between_3-ERA5_LAND_and_1-C3S']

    def test_iter_metrics(self):
        for Metr in self.img._iter_metrics(**{'g':2}):
            assert Metr.name in globals.metric_groups[2]

    def test_group_vars(self):
        Vars = self.img.group_vars(**{'metric':'R'})
        names = [Var.varname for Var in Vars]
        assert names == ['R_between_3-ERA5_LAND_and_1-C3S', 'R_between_3-ERA5_LAND_and_2-SMOS']

    def test_group_metrics(self):
        common, double, triple = self.img.group_metrics(['R'])
        assert common == {}
        assert triple == {}
        assert list(double.keys()) == ['R']

    def test_load_metrics(self):
        assert len(self.img.metrics.keys()) == len(globals.metric_groups[0]) + len(globals.metric_groups[2])

    def test_ds2df(self):
        df = self.img._ds2df()
        assert len(df.columns) == len(self.img.varnames) - 2  # minus lon, lat

    def test_metric_df(self):
        df = self.img.metric_df(['R'])
        assert list(df.columns) == ['R_between_3-ERA5_LAND_and_1-C3S', 'R_between_3-ERA5_LAND_and_2-SMOS']

    def test_metrics_in_file(self):
        """Test that all metrics are initialized correctly"""
        assert list(self.img.common.keys()) == globals.metric_groups[0]
        for m in self.img.double.keys():  # tau is not in the results
            assert m in globals.metric_groups[2]
        assert list(self.img.triple.keys()) == []  #  this is not the TC test case

        # with merged return value
        ms = self.img.metrics
        for m in ms:
            assert any([m in l for l in list(globals.metric_groups.values())])

    def test_vars_in_file(self):
        """Test that all variables are initialized correctly"""
        vars = []
        for Var in self.img._iter_vars(only_metrics=True):
            vars.append(Var.varname)
        vars_should = ['n_obs']
        # since the valination is non-TC
        for metric in globals.metric_groups[2]:
            vars_should.append('{}_between_3-ERA5_LAND_and_1-C3S'.format(metric))
            vars_should.append('{}_between_3-ERA5_LAND_and_2-SMOS'.format(metric))
        vars_should = np.sort(np.array(vars_should))
        vars = np.sort(np.array(vars))

        assert all(vars == vars_should)

    def test_find_groups(self):
        """Test that all metrics for a specific group can be collected"""
        common_group = []
        for name, Metric in self.img.common.items():
            assert Metric.name in globals.metric_groups[0]
            assert len(Metric.variables) == 1
            common_group.append(name)
        double_group =[]
        for name, Metric in self.img.double.items():
            assert Metric.name in globals.metric_groups[2]
            assert len(Metric.variables) == 2
            double_group.append(name)

        assert self.img.triple == {}

    def test_variable_datasets(self):
        """Test the metadata associated with the ref dataset of the double group variables"""
        for Var in self.img._iter_vars(**{'g':2}):
            ref_ds, metric_ds, other_ds = Var.get_varmeta()
            assert ref_ds[1]['short_name'] == 'ERA5_LAND'
            assert ref_ds[1]['pretty_name'] == 'ERA5-Land'
            assert other_ds is None

    def test_ref_meta(self):
        """Test the metadata associated with the ref dataset of the image"""
        ref_meta = self.img.datasets.ref
        assert ref_meta['short_name'] == 'ERA5_LAND'
        assert ref_meta['pretty_name'] == 'ERA5-Land'
        assert ref_meta['short_version'] == 'ERA5_LAND_V20190904'
        assert ref_meta['pretty_version'] == 'v20190904'
        assert ref_meta['pretty_title'] == 'ERA5-Land (v20190904)'

    def test_var_meta(self):
        """Test datasets associated with a specific variable"""
        for Var in self.img._iter_vars(**{'varname':'R_between_3-ERA5_LAND_and_1-C3S'}):
            ref_id, ref_meta = Var.ref_ds
            assert ref_id == 3
            assert ref_meta['short_name'] == 'ERA5_LAND'
            assert ref_meta['pretty_name'] == 'ERA5-Land'
            assert ref_meta['pretty_version'] == 'v20190904'

            metric_id, metric_meta = Var.metric_ds
            assert metric_id == 1
            assert metric_meta['short_name'] == 'C3S'
            assert metric_meta['pretty_name'] == 'C3S'
            assert metric_meta['pretty_version'] == 'v201812'
    
    def test_metric_stats(self):
        """Test the function metric_stats"""
        for name, Metric in self.img.metrics.items():
            stats = self.img._metric_stats(name)
            group = Metric.g
            if stats:  # empty variables return an empty list
                if group == 0:
                    assert len(stats) == 1
                elif group == 2:
                    assert len(stats) == 2
    
    def test_stats_df(self):
        """Test the stats dataframe"""
        df = self.img.stats_df()
        empty_metrics = 0
        for name, Metric in self.img.metrics.items():
            stats = self.img._metric_stats(name)
            if not stats:  # find metrics without values
                if Metric.g == 1:
                    empty_metrics += 1
                elif Metric.g == 2:  # stats table has an entry for metric, for sat dataset (in common and triple metrics)
                    empty_metrics += 2

        tot_stats = len(self.img.common.keys()) + 2*len(self.img.double.keys()) - empty_metrics
        assert tot_stats == len(df)


class TestQA4SMImgWithCI(unittest.TestCase): # todo: update with correct CI .nc file
    """Test image where some of the variables are confidence intervals"""

    def setUp(self) -> None:
        self.testfile = "0-ERA5.swvl1_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_combined.sm_with_3-ESA_CCI_SM_combined.sm_with_4-ESA_CCI_SM_combined.sm.CI.nc"
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'tc', self.testfile)
        self.img = QA4SMImg(self.testfile_path, ignore_empty=False)

    def test_testfile(self):
        someCIs = [
            "RMSD_ci_lower_between_0-ERA5_and_1-ESA_CCI_SM_combined",
            "RMSD_ci_upper_between_0-ERA5_and_1-ESA_CCI_SM_combined"
        ]
        for CI_varname in someCIs:
            assert CI_varname in self.img.varnames

    def test_CIs(self):
        assert self.img.has_CIs

    def test_CI_in_Vars(self):
        """Test that CI Variables are correctly assigned to a metric"""
        for CI_varname in self.img._iter_vars(**{
            "metric":"RMSD",
            "metric_ds":"2-ESA_CCI_SM_combined"}):
            assert CI_varname in [
                "RMSD_ci_lower_between_0-ERA5_and_2-ESA_CCI_SM_combined",
                "RMSD_ci_upper_between_0-ERA5_and_2-ESA_CCI_SM_combined"
            ]

if __name__ == '__main__':
    unittest.main()
