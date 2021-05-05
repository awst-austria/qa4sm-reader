# -*- coding: utf-8 -*-
import pandas as pd
import unittest

from qa4sm_reader.handlers import QA4SMDatasets, QA4SMMetricVariable, QA4SMMetric

from tests.test_attr import test_attributes, test_tc_attributes, test_CI_attributes


class TestQA4SMDatasets(unittest.TestCase):
    """Tests based on netCDF file where reference id == 6 (i.e. different from 0)"""

    def setUp(self) -> None:
        attrs = test_attributes()
        self.Datasets = QA4SMDatasets(attrs)
        self.ismn = self.Datasets._dc_names(dc=5)
        self.c3s17 = self.Datasets._dc_names(dc=0)
        self.c3s18 = self.Datasets._dc_names(dc=1)
        self.smos = self.Datasets._dc_names(dc=2)
        self.smap = self.Datasets._dc_names(dc=3)
        self.ascat = self.Datasets._dc_names(dc=4)

    def test_id_dc(self):
        assert self.Datasets._ref_dc() != self.Datasets._ref_id()
        assert self.Datasets._ref_id() == 6
        assert self.Datasets.offset == -1
        assert self.Datasets._id2dc(6) == 5

    def test_dcs(self):
        for i in range(5):
            assert i in self.Datasets._dcs().keys()
        assert len(self.Datasets._dcs().keys()) == 5

    def test_dc_names(self):
        assert self.ismn['pretty_name'] == 'ISMN'
        assert self.ismn['pretty_version'] == '20180712 mini testset'

        assert self.c3s17['pretty_name'] == 'C3S'
        assert self.c3s17['pretty_version'] == 'v201706'

        assert self.c3s18['pretty_name'] == 'C3S'
        assert self.c3s18['pretty_version'] == 'v201812'

        assert self.smos['pretty_name'] == 'SMOS IC'
        assert self.smos['pretty_version'] == 'V.105 Ascending'

        assert self.smap['pretty_name'] == 'SMAP level 3'
        assert self.smap['pretty_version'] == 'v5 PM/ascending'

        assert self.ascat['pretty_name'] == 'H-SAF ASCAT SSM CDR'
        assert self.ascat['pretty_version'] == 'H113'

    def test_others(self):
        assert len(self.Datasets.others) == 5

    def test_dataset_metadata(self):
        meta_ref = self.Datasets.dataset_metadata(6)[1]  # shape (id, {names})
        also_meta_ref = self.Datasets._dc_names(5)
        assert meta_ref == also_meta_ref

class TestMetricVariableTC(unittest.TestCase):

    def setUp(self) -> None:
        attrs = test_tc_attributes()
        df_nobs = pd.DataFrame(index=range(10), data={'n_obs': range(10)})
        self.n_obs = QA4SMMetricVariable('n_obs', attrs, values=df_nobs)
        self.r = QA4SMMetricVariable('R_between_3-ERA5_LAND_and_1-C3S', attrs)
        self.beta = QA4SMMetricVariable('beta_1-C3S_between_3-ERA5_LAND_and_1-C3S_and_2-ASCAT', attrs)

    def test_properties(self):
        assert self.beta.isempty
        assert self.beta.ismetric

    def test_pretty_name(self):
        assert self.beta.pretty_name == "TC scaling coefficient of C3S (v201812) \n against ERA5-Land (ERA5-Land test), H-SAF ASCAT SSM CDR (H113)"

    def test_parse_varname(self):
        for var in [self.beta, self.r, self.n_obs]:
            info = var._parse_varname()
            assert type(info[0]) == str
            assert type(info[1]) == int
            assert type(info[2]) == dict

    def test_get_varmeta(self):
        # n_obs has only the reference dataset
        assert self.n_obs.ismetric
        assert not self.n_obs.isempty
        ref_ds, metric_ds, other_ds = self.n_obs.get_varmeta()
        assert ref_ds[1]['short_name'] == 'ERA5_LAND'
        assert metric_ds == other_ds is None

        # R has only the reference and metric dataset
        ref_ds, metric_ds, other_ds = self.r.get_varmeta()
        assert ref_ds[0] == 3
        assert ref_ds[1]['short_name'] == 'ERA5_LAND'
        assert ref_ds[1]['pretty_name'] == 'ERA5-Land'
        assert ref_ds[1]['short_version'] == 'ERA5_LAND_TEST'
        assert ref_ds[1]['pretty_version'] == 'ERA5-Land test'

        assert metric_ds[0] == 1
        mds_meta = metric_ds[1]
        assert mds_meta['short_name'] == 'C3S'
        assert mds_meta['pretty_name'] == 'C3S'
        assert mds_meta['short_version'] == 'C3S_V201812'
        assert mds_meta['pretty_version'] == 'v201812'
        assert other_ds is None

        # p has all three datasets, it being a TC metric
        ref_ds, metric_ds, other_ds = self.beta.get_varmeta()
        assert ref_ds[0] == 3
        assert ref_ds[1]['short_name'] == 'ERA5_LAND'
        assert ref_ds[1]['pretty_name'] == 'ERA5-Land'
        assert ref_ds[1]['short_version'] == 'ERA5_LAND_TEST'
        assert ref_ds[1]['pretty_version'] == 'ERA5-Land test'

        assert metric_ds[0] == 1
        assert other_ds[0] == 2
        mds_meta = metric_ds[1]
        other_meta = other_ds[1]
        assert mds_meta['short_name'] == 'C3S'
        assert mds_meta['pretty_name'] == 'C3S'
        assert mds_meta['short_version'] == 'C3S_V201812'
        assert mds_meta['pretty_version'] == 'v201812'

        assert other_meta['short_name'] == 'ASCAT'
        assert other_meta['pretty_name'] == 'H-SAF ASCAT SSM CDR'
        assert other_meta['short_version'] == 'ASCAT_H113'
        assert other_meta['pretty_version'] == 'H113'


class TestMetricVariableBasic(unittest.TestCase):

    def setUp(self) -> None:
        attrs = test_attributes()
        df_nobs = pd.DataFrame(index=range(10), data={'n_obs': range(10)})
        self.n_obs = QA4SMMetricVariable('n_obs', attrs, values=df_nobs)

        self.r = QA4SMMetricVariable('R_between_6-ISMN_and_4-SMAP', attrs)
        self.pr = QA4SMMetricVariable('p_rho_between_6-ISMN_and_5-ASCAT', attrs)

    def test_get_varmeta(self):
        # n_obs
        assert self.n_obs.ismetric
        assert not self.n_obs.isempty
        ref_ds, metric_ds, other_ds = self.n_obs.get_varmeta()
        assert ref_ds[1]['short_name'] == 'ISMN'
        assert metric_ds == other_ds is None

        # R
        ref_ds, metric_ds, other_ds = self.r.get_varmeta()
        assert ref_ds[0] == 6
        assert ref_ds[1]['short_name'] == 'ISMN'
        assert ref_ds[1]['pretty_name'] == 'ISMN'
        assert ref_ds[1]['short_version'] == 'ISMN_V20180712_MINI'
        assert ref_ds[1]['pretty_version'] == '20180712 mini testset'
        assert metric_ds[0] == 4
        mds_meta = metric_ds[1]
        assert mds_meta['short_name'] == 'SMAP'
        assert mds_meta['pretty_name'] == 'SMAP level 3'
        assert mds_meta['short_version'] == 'SMAP_V5_PM'
        assert mds_meta['pretty_version'] == 'v5 PM/ascending'
        assert other_ds is None

        # p
        ref_ds, metric_ds, other_ds = self.pr.get_varmeta()
        assert ref_ds[0] == 6
        assert ref_ds[1]['short_name'] == 'ISMN'
        assert ref_ds[1]['pretty_name'] == 'ISMN'
        assert ref_ds[1]['short_version'] == 'ISMN_V20180712_MINI'
        assert ref_ds[1]['pretty_version'] == '20180712 mini testset'
        assert metric_ds[0] == 5
        mds_meta = metric_ds[1]
        assert mds_meta['short_name'] == 'ASCAT'
        assert mds_meta['pretty_name'] == 'H-SAF ASCAT SSM CDR'
        assert mds_meta['short_version'] == 'ASCAT_H113'
        assert mds_meta['pretty_version'] == 'H113'
        assert other_ds is None


class TestQA4SMMetric(unittest.TestCase):

    def setUp(self) -> None:
        attrs = test_tc_attributes()

        self.r1 = QA4SMMetricVariable('R_between_3-ERA5_LAND_and_2-ASCAT', attrs)
        self.r2 = QA4SMMetricVariable('R_between_3-ERA5_LAND_and_1-C3S', attrs)
        self.R = QA4SMMetric('R', variables_list=[self.r1, self.r2])

    def test_get_attribute(self):
        assert self.R.g == self.r1.g == self.r2.g


class TestMetricVariableCI(unittest.TestCase): # todo: update with correct CI .nc file
    """Test variables in image with confidence intervals"""
    def setUp(self) -> None:
        attrs = test_CI_attributes()
        self.CI_Var = QA4SMMetricVariable(
            "RMSD_ci_upper_between_0-ERA5_and_2-ESA_CCI_SM_combined",
            attrs
        )

    def test_CI_var(self):
        assert  self.CI_Var.ismetric
        assert self.CI_Var.is_CI
        print(self.CI_Var.pretty_name)
        assert self.CI_Var.pretty_name == "Confidence Interval of Root-mean-square deviation of ESA CCI " \
                                          "SM combined (v05.2) \n with ERA5 (v20190613) as reference"
        assert self.CI_Var.bound == "upper"

if __name__ == '__main__':
    unittest.main()
