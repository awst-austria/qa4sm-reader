import pytest
import tempfile
import os

from qa4sm_reader.custom_user_plot_generator import (
    CustomPlotObject
)

@pytest.fixture
def plotdir():
    plotdir = tempfile.mkdtemp()

    return plotdir



@pytest.fixture
def sample_plot_object():
    """Test combined_boxplot with valid input."""
    file_name = '3-ERA5_LAND.swvl1_with_1-C3S.sm_with_2-ASCAT.sm.nc'
    file_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                   'test_data', 'tc', file_name)
    custom_plot_object = CustomPlotObject(file_path)
    return custom_plot_object


def test_display_variables(sample_plot_object):
    """Test plot object creation."""
    try :
        sample_plot_object.display_metrics_and_datasets()
    except Exception as e:
        pytest.fail(f"display_variables raised an exception {e}")


def test_plot_map(sample_plot_object):
    '''Test plot object creation.'''
    metric = 'R'
    temp_dir = tempfile.mkdtemp()
    dataset_names = ['ERA5_LAND', 'C3S']
    sample_plot_object.plot_map(metric, temp_dir, dataset_list=dataset_names)
    assert os.path.exists(
        os.path.join(temp_dir, "R_between_3-ERA5_LAND_and_1-C3S_map.png"))









