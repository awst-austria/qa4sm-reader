# -*- coding: utf-8 -*-

from qa4sm_reader.plotter import QA4SMPlotter
from qa4sm_reader.img import QA4SMImg
from qa4sm_reader.plot_utils import geotraj_to_geo2d
import os
import unittest
import tempfile
import shutil


class TestQA4SMMetaImgISMNPlotter_newFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.testfile = '0-ISMN.soil moisture_with_1-C3S.sm.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'basic', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot_metric('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot_metric('R', out_types='svg', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.mapplot_metric('BIAS', out_types='png', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)  # cleanup

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_types='svg', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_types='png', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)


class TestQA4SMMetaImgGLDASPlotter_newFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.testfile = '0-GLDAS.SoilMoi0_10cm_inst_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'basic', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot_metric('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot_metric('R', out_types='svg', save_files=True) # should be 2 files
        assert len(os.listdir(self.plotdir)) == 1 + 2
        assert len(list(r_files)) == 2

        bias_files = self.plotter.mapplot_metric('BIAS', out_types='png', save_files=True) # should be 2 files
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2
        assert len(list(bias_files)) == 2

        shutil.rmtree(self.plotdir)

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_types='svg', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_types='png', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)


class TestQA4SMMetaImgBasicPlotter(unittest.TestCase):

    def setUp(self) -> None:
        self.testfile = '3-GLDAS.SoilMoi0_10cm_inst_with_1-C3S.sm_with_2-SMOS.Soil_Moisture.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'tc', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot_metric('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot_metric('R', out_types='svg', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2
        assert len(list(r_files)) == 2

        bias_files = self.plotter.mapplot_metric('BIAS', out_types='png', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2
        assert len(list(bias_files)) == 2


        snr_files = self.plotter.mapplot_metric('snr', out_types='svg', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2 + 2
        assert len(list(snr_files)) == 2

        err_files = self.plotter.mapplot_metric('err_std', out_types='svg', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 2 + 2 + 2 + 2
        assert len(list(err_files)) == 2

        shutil.rmtree(self.plotdir)

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_types='svg', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_types='png', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        snr_files = self.plotter.boxplot_tc('snr', out_types='svg', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1 + 2
        assert len(list(snr_files)) == 2

        err_files = self.plotter.boxplot_tc('err_std', out_types='svg', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1 + 2 + 2
        assert len(list(err_files)) == 2

        shutil.rmtree(self.plotdir)

class TestQA4SMMetaImgIrregularGridPlotter(unittest.TestCase):
    def setUp(self) -> None:
        self.testfile = '0-SMAP.soil_moisture_with_1-C3S.sm.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                          'test_data', 'basic', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)
        self.ref_dataset_grid_stepsize = self.img.ref_dataset_grid_stepsize

    # def test_netCDF_file:
    #     opened_file =

    def test_mapplot(self):
        n_obs_files = self.plotter.mapplot_metric('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.mapplot_metric('R', out_types='svg', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.mapplot_metric('BIAS', out_types='png', save_files=True) # should be 2
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)

    def test_boxplot(self):
        n_obs_files = self.plotter.boxplot_basic('n_obs', out_types='png', save_files=True) # should be 1
        assert len(list(n_obs_files)) == 1
        assert len(os.listdir(self.plotdir)) == 1

        r_files = self.plotter.boxplot_basic('R', out_types='svg', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1
        assert len(list(r_files)) == 1

        bias_files = self.plotter.boxplot_basic('BIAS', out_types='png', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1 + 1 + 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)

    def test_grid_creation(self):
        metric = 'n_obs'
        for Var in self.img._iter_vars(**{'metric':metric}):
            varname = Var.varname
            df = self.img._ds2df([varname])[varname]
            zz, grid, origin = geotraj_to_geo2d(df, grid_stepsize=self.ref_dataset_grid_stepsize)
            print('varname: ', varname, 'zz: ', zz, 'grid: ', grid)
            assert zz.count() != 0
            assert origin == 'upper'


class TestQA4SMMetaImgWithCIPlotter(unittest.TestCase):
    """Test plotter works with confidence intervals. We test whether boxplots are working, but not whether CIs are
    actually plotted correctly inside them. Current CI file includes bootstrapped intervals"""

    def setUp(self) -> None:
        self.testfile = "0-ERA5.swvl1_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_combined.sm_with_3-ESA_CCI_SM_combined.sm_with_4-ESA_CCI_SM_combined.sm.CI.nc"
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                          'test_data', 'tc', self.testfile)
        self.plotdir = tempfile.mkdtemp()
        self.img = QA4SMImg(self.testfile_path)
        self.plotter = QA4SMPlotter(self.img, self.plotdir)

    def test_boxplot_basic(self):
        bias_files = self.plotter.boxplot_basic('BIAS', out_types='png', save_files=True) # should be 1
        assert len(os.listdir(self.plotdir)) == 1
        assert len(list(bias_files)) == 1

        shutil.rmtree(self.plotdir)

    def test_boxplot_tc(self):
        snr_files = self.plotter.boxplot_tc('snr', out_types='svg', save_files=True) # should be 4
        assert len(os.listdir(self.plotdir)) == 4
        assert len(list(snr_files)) == 4

        shutil.rmtree(self.plotdir)

if __name__ == '__main__':
    unittest.main()
