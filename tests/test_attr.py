# -*- coding: utf-8 -*-

from qa4sm_reader.handlers import QA4SMDatasets, QA4SMMetricVariable, QA4SMMetric
import os
import unittest
import xarray as xr


def test_attributes():
    testfile = os.path.join(os.path.dirname(__file__), 'test_data', 'basic',
        '6-ISMN.soil moisture_with_1-C3S.sm_with_2-C3S.sm_with_3-SMOS.Soil_Moisture_with_4-SMAP.soil_moisture_with_5-ASCAT.sm.nc')
    ds = xr.open_dataset(testfile)
    return ds.attrs

def test_tc_attributes():
    testfile = os.path.join(os.path.dirname(__file__), 'test_data', 'tc',
                            '3-ERA5_LAND.swvl1_with_1-C3S.sm_with_2-ASCAT.sm.nc')
    ds = xr.open_dataset(testfile)
    return ds.attrs

def test_CI_attributes():
    testfile = os.path.join(os.path.dirname(__file__), 'test_data', 'tc',
                            "0-ERA5.swvl1_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_combined.sm_with_3-ESA_CCI_SM_combined.sm_with_4-ESA_CCI_SM_combined.sm.CI.nc")
    ds = xr.open_dataset(testfile)
    return ds.attrs

def test_grid_stepsize():
    testfile = os.path.join(os.path.dirname(__file__), 'test_data', 'basic',
                            '0-SMAP.soil_moisture_with_1-C3S.sm.nc')
    ds = xr.open_dataset(testfile)
    return ds.attrs

class TestQA4SMAttributes(unittest.TestCase):

    def setUp(self) -> None:
        attrs = test_attributes()
        self.meta = QA4SMDatasets(global_attrs=attrs)

    def test_get_ref_name(self):
        ref_names = self.meta.ref
        assert ref_names['short_name'] == 'ISMN'
        assert ref_names['pretty_name'] == 'ISMN'
        assert ref_names['short_version'] == 'ISMN_V20180712_MINI'
        assert ref_names['pretty_version'] == '20180712 mini testset'
        return ref_names

    def test_get_other_names(self):
        other_names = self.meta.others
        # index is dc, as in the meta values not as in the variable name
        assert other_names[0]['short_name'] == 'C3S'
        assert other_names[0]['pretty_name'] == 'C3S'
        assert other_names[0]['short_version'] == 'C3S_V201706'
        assert other_names[0]['pretty_version'] == 'v201706'

        assert other_names[1]['short_name'] == 'C3S'
        assert other_names[1]['pretty_name'] == 'C3S'
        assert other_names[1]['short_version'] == 'C3S_V201812'
        assert other_names[1]['pretty_version'] == 'v201812'

        assert other_names[2]['short_name'] == 'SMOS'
        assert other_names[2]['pretty_name'] == 'SMOS IC'
        assert other_names[2]['short_version'] == 'SMOS_105_ASC'
        assert other_names[2]['pretty_version'] == 'V.105 Ascending'

        assert other_names[3]['short_name'] == 'SMAP'
        assert other_names[3]['pretty_name'] == 'SMAP level 3'
        assert other_names[3]['short_version'] == 'SMAP_V5_PM'
        assert other_names[3]['pretty_version'] == 'v5 PM/ascending'

        assert other_names[4]['short_name'] == 'ASCAT'
        assert other_names[4]['pretty_name'] == 'H-SAF ASCAT SSM CDR'
        assert other_names[4]['short_version'] == 'ASCAT_H113'
        assert other_names[4]['pretty_version'] == 'H113'

        return other_names

    def test_grid_stepsize(self):
        attrs = test_grid_stepsize()
        assert attrs['val_dc_dataset0_grid_stepsize'] == 0.35


if __name__ == '__main__':
    unittest.main()
