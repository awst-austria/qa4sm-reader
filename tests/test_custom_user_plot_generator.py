import shutil

import pytest
import tempfile
import os

from qa4sm_reader.custom_user_plot_generator import (
    CustomPlotObject
)

@pytest.fixture
def plotdir():
    # Create a temporary directory
    dir_path = tempfile.mkdtemp()
    yield dir_path   # Provide the directory to the test
    # Cleanup after test
    shutil.rmtree(dir_path)

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

def test_plot_map_basic(sample_plot_object, plotdir):
    try:
        sample_plot_object.plot_map(metric='R', output_dir=plotdir, dataset_list=['ERA5_LAND', 'C3S'])
    except Exception as e:
        pytest.fail(f"plot_map raised an exception {e}")

def test_plot_map_cmap_value_range(sample_plot_object, plotdir):
    try:
        sample_plot_object.plot_map(metric='R', output_dir=plotdir, dataset_list=['ERA5_LAND', 'C3S'], colormap="viridis", value_range=(-0.5, 0.5))
    except Exception as e:
        pytest.fail(f"plot_map raised an exception {e}")
def test_plot_map_cmap_extent(sample_plot_object, plotdir):
    try:
        sample_plot_object.plot_map(metric='R', output_dir=plotdir, dataset_list=['ERA5_LAND', 'C3S'], colormap="viridis", extent=(-10, 20, 30, 40))
    except Exception as e:
        pytest.fail(f"plot_map raised an exception {e}")
def test_plot_map_cmap_extent(sample_plot_object, plotdir):
    try:
        sample_plot_object.plot_map(metric='R', output_dir=plotdir, dataset_list=['ERA5_LAND', 'C3S'], colormap="viridis", extent=(-10, 20, 30, 40))
    except Exception as e:
        pytest.fail(f"plot_map raised an exception {e}")
def test_plot_map_title_fontsizes_label(sample_plot_object, plotdir):
    try:
        sample_plot_object.plot_map(metric='R', output_dir=plotdir, dataset_list=['ERA5_LAND', 'C3S'], title='Pearson Correlation between ERA5_Land and C3S_combined', title_fontsize=20,  colorbar_label='Pearson Correlation Coefficient', colorbar_ticks_fontsize=20,
            xy_ticks_fontsize=20)
    except Exception as e:
        pytest.fail(f"plot_map raised an exception {e}")

def test_plot_map_plotsize(sample_plot_object, plotdir):
    try:
        sample_plot_object.plot_map(metric='R', output_dir=plotdir, dataset_list=['ERA5_LAND', 'C3S'], plotsize=(20, 10))
    except Exception as e:
        pytest.fail(f"plot_map raised an exception {e}")
def test_plot_map_triple_col(sample_plot_object, plotdir):
    try:
        sample_plot_object.plot_map(metric='snr', output_dir=plotdir, dataset_list=['C3S'], title='ISMN SNR relative to ERA5_LAND and ESA_CCI_SM_passive', title_fontsize=20, colorbar_label='SNR', colorbar_ticks_fontsize=20, xy_ticks_fontsize=20)
    except Exception as e:
        pytest.fail(f"plot_map raised an exception {e}")





