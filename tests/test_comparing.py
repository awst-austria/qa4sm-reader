from qa4sm_reader.comparing import QA4SMComparison
from qa4sm_reader.img import QA4SMImg

import os
import unittest

import pandas as pd
import matplotlib.pyplot as plt


class TestQA4SMComparison_SingleImage(unittest.TestCase):
    """Test cases where a single validation is initialized, with two satellite datasets"""

    def setUp(self) -> None:
        self.testfile = '3-ERA5_LAND.swvl1_with_1-C3S.sm_with_2-ASCAT.sm.nc'
        self.testfile_path = os.path.join(os.path.dirname(__file__), '..','tests',
                                          'test_data', 'tc', self.testfile)
        self.comp = QA4SMComparison(self.testfile_path)

    def test_init(self):
        assert isinstance(self.comp.comparison, QA4SMImg)

    def test_check_ref(self):
        assert self.comp._check_ref() == {
            'short_name': 'ERA5_LAND',
            'pretty_name': 'ERA5-Land',
            'short_version': 'ERA5_LAND_TEST',
            'pretty_version': 'ERA5-Land test',
            'pretty_title': 'ERA5-Land (ERA5-Land test)'
        }

    def test_checks(self):
        assert self.comp.perform_checks()

    def test_get_pairwise(self):
        pair = self.comp._get_pairwise("R")

        assert isinstance(pair, pd.DataFrame)
        assert len(pair.columns) == 3, "There should be one column for comparison term" \
                                       "plus the column with difference values"

    def test_wrapper(self):
        """
        This tests the wrapper function but more in general also the
        plotting functions/table
        """
        methods = [
            'table',
            'boxplot',
            'correlation',
            'difference',
            'mapplot'
        ]
        for method in methods:
            out = self.comp.wrapper(method, "R")
            plt.close("all")
            if method == "table":
                assert out is not None  # generates a pandas dataframe
            else:
                assert not out  # generates a plot and returns nothing

class TestQA4SMComparison_DoubleOverlapping(unittest.TestCase):
    """Test cases where two partially overlapping validations are initilized"""

    def setUp(self) -> None:
        self.first = '0-ISMN.soil moisture_with_1-C3S.sm.nc'
        self.second = '0-ISMN.soil moisture_with_1-C3S.sm-overlap.nc'
        self.testfile_paths = [
            os.path.join(
                os.path.dirname(__file__),
                '..','tests','test_data', 'comparing',
                i
            ) for i in [
                self.first,
                self.second]
        ]
        # initialized with intersection
        self.comp = QA4SMComparison(self.testfile_paths)

    def test_intersection(self):
        assert not self.comp.union

    def test_init(self):
        assert [
            i for i in self.comp.comparison.keys()
        ] == self.testfile_paths

    def test_geometry(self):
        assert self.comp.get_extent() != self.comp.get_extent(get_intersection=False)

    def test_get_pairwise(self):
        pair = self.comp._get_pairwise("R")

        assert isinstance(pair, pd.DataFrame)
        assert len(pair.columns) == 3, "There should be one column for comparison term" \
                                       "plus the column with difference values"

    def test_checks(self):
        """No assertion, but will throw error if any of the checks are not passed"""
        self.comp.perform_checks(
            overlapping=True,
            union=True,
            pairwise=True
        )

    def test_wrapper(self):
        """
        This tests the wrapper function but more in general also the
        plotting functions/table
        """
        methods = [
            'table',
            'boxplot',
            'correlation',
            'difference',
            'mapplot'
        ]
        for method in methods:
            out = self.comp.wrapper(method, "R")
            plt.close("all")
            if method == "table":
                assert out is not None  # generates a pandas dataframe
            else:
                assert not out  # generates a plot and returns nothing

    def test_init_union(self):
        """Should go at the end as it chnages the attributes"""
        self.comp.init_union()
        assert self.comp.union

    def test_pairwise_methods(self):
        comp = QA4SMComparison(
            self.testfile_paths,
            get_intersection=False
        )  # todo: solve unexpected behavior on perform_checks
        works = False
        methods = [
            'correlation',
            'difference',
            'mapplot'
        ]
        for method in methods:
            try:  # they all have same behavior
                comp.wrapper(
                    method,
                    metric="R"
                )
            except AssertionError:
                works = True

        assert works


class TestQA4SMComparison_DoubleNonOverlapping(unittest.TestCase):
    """Test cases where two non overlapping validations are initilized"""

    def setUp(self) -> None:
        self.first = '0-ISMN.soil moisture_with_1-C3S.sm.nc'
        self.second = '0-ISMN.soil moisture_with_1-C3S.sm-nonoverlap.nc'
        self.testfile_paths = [
            os.path.join(
                os.path.dirname(__file__),
                '..','tests','test_data', 'comparing',
                i
            ) for i in [
                self.first,
                self.second]
        ]
        self.comp = QA4SMComparison(
            self.testfile_paths,
            get_intersection=False
        )

    def test_init(self):
        works = False
        try:
            comp = QA4SMComparison(
                self.testfile_paths
            )
        except AssertionError:
            works = True

        assert works

    def test_check_initialized(self):
        for path in self.comp.comparison.keys():
            self.comp._check_initialized(path)

    # todo: tests for initialized different extent

if __name__ == '__main__':
    unittest.main()