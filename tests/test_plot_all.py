# test for functions that plot all the images. Use pytest.long_run to avoid running it for development
import os
import sys

import pytest
import tempfile
import shutil
from pathlib import Path

from qa4sm_reader.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber
import qa4sm_reader.plot_all as pa
from qa4sm_reader.utils import transcribe

# if sys.platform.startswith("win"):
#     pytestmark = pytest.mark.skip(
#         "Failing on Windows."
#     )


@pytest.fixture
def plotdir():
    plotdir = tempfile.mkdtemp()

    return plotdir


def test_plot_all(plotdir):
    """Plot all - including metadata based plots - to temporary directory and count files"""
    testfile = '0-ISMN.soil_moisture_with_1-C3S.sm.nc'
    testfile_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                                 'test_data', 'metadata', testfile)

    temporal_sub_windows_present = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(testfile_path)
    if not temporal_sub_windows_present:
        dataset = transcribe(testfile_path)

        tmp_testfile_path = Path(plotdir + '/tmp_testfile.nc')
        encoding={var: {'zlib': False} for var in dataset.variables}
        dataset.to_netcdf(tmp_testfile_path, encoding=encoding)
        testfile_path = tmp_testfile_path
        temporal_sub_windows_present = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(testfile_path)


    pa.plot_all(
        filepath=testfile_path,
        temporal_sub_windows=temporal_sub_windows_present,
        out_dir=plotdir,
        save_all=True,
        save_metadata=True,
    )

    for tswp in temporal_sub_windows_present:
        assert len(os.listdir(os.path.join(plotdir, tswp))) == 71
        assert all(os.path.splitext(file)[1] in [".png", ".csv"] for file in os.listdir(os.path.join(plotdir, tswp))), \
            "Not all files have been saved as .png or .csv"

    shutil.rmtree(plotdir)
