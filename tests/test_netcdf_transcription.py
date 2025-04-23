import pytest
from copy import deepcopy
from datetime import datetime
import xarray as xr
import shutil
from pathlib import Path
from typing import Union, Optional, Tuple, List
import logging
import numpy as np
import tempfile

from qa4sm_reader.netcdf_transcription import Pytesmo2Qa4smResultsTranscriber, TemporalSubWindowMismatchError
from qa4sm_reader.intra_annual_temp_windows import TemporalSubWindowsCreator, NewSubWindow, InvalidTemporalSubWindowError, TemporalSubWindowsFactory
import qa4sm_reader.globals as globals
from qa4sm_reader.utils import log_function_call
import qa4sm_reader.plot_all as pa

log_file_path = Path(
    __file__).parent.parent / '.logs' / "test_netcdf_transcription.log"
if not log_file_path.parent.exists():
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=str(log_file_path))


#-------------------------------------------------------Fixtures--------------------------------------------------------
@pytest.fixture(scope="module")
def tmp_paths():
    '''Fixture to keep track of temporary directories created during a test run and clean them up after the test run'''
    paths = []
    yield paths

    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def monthly_tsws() -> TemporalSubWindowsCreator:
    return TemporalSubWindowsCreator(temporal_sub_window_type='months',
                                     overlap=0,
                                     custom_file=None)


@pytest.fixture
def monthly_tsws_incl_bulk(monthly_tsws) -> TemporalSubWindowsCreator:
    bulk_wndw = NewSubWindow('bulk', datetime(1950, 1, 1),
                             datetime(2020, 1, 1))
    return monthly_tsws.add_temp_sub_wndw(bulk_wndw, insert_as_first_wndw=True)


@pytest.fixture
def seasonal_tsws() -> TemporalSubWindowsCreator:
    return TemporalSubWindowsCreator(temporal_sub_window_type='seasons',
                                     overlap=0,
                                     custom_file=None)


@pytest.fixture
def seasonal_tsws_incl_bulk() -> TemporalSubWindowsCreator:
    seasonal_tsws = TemporalSubWindowsCreator(
        temporal_sub_window_type='seasons', overlap=0, custom_file=None)
    bulk_wndw = NewSubWindow('bulk', datetime(1950, 1, 1),
                             datetime(2020, 1, 1))
    seasonal_tsws.add_temp_sub_wndw(bulk_wndw, insert_as_first_wndw=True)
    return seasonal_tsws


@pytest.fixture
def seasonal_pytesmo_file(TEST_DATA_DIR) -> Path:
    return Path(
        TEST_DATA_DIR / 'intra_annual' / 'seasonal' /
        '0-ERA5.swvl1_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_combined.sm_with_3-ESA_CCI_SM_combined.sm_with_4-ESA_CCI_SM_combined.sm.CI_tsw_seasons_pytesmo.nc'
    )


@pytest.fixture
def seasonal_qa4sm_file(TEST_DATA_DIR) -> Path:
    return Path(
        TEST_DATA_DIR / 'intra_annual' / 'seasonal' /
        '0-ERA5.swvl1_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_combined.sm_with_3-ESA_CCI_SM_combined.sm_with_4-ESA_CCI_SM_combined.sm.CI_tsw_seasons_qa4sm.nc'
    )


@pytest.fixture
def monthly_pytesmo_file(TEST_DATA_DIR) -> Path:
    return Path(TEST_DATA_DIR / 'intra_annual' / 'monthly' /
                '0-ISMN.soil_moisture_with_1-C3S.sm_tsw_months_pytesmo.nc')


@pytest.fixture
def monthly_qa4sm_file(TEST_DATA_DIR) -> Path:
    return Path(TEST_DATA_DIR / 'intra_annual' / 'monthly' /
                '0-ISMN.soil_moisture_with_1-C3S.sm_tsw_months_qa4sm.nc')

@pytest.fixture
def stability_pytesmo_file(TEST_DATA_DIR) -> Path:
    return Path(TEST_DATA_DIR / 'intra_annual' / 'stability' / '0-ESA_CCI_SM_passive.sm_with_1-ERA5_LAND.swvl1_tsw_stability_pytesmo.nc')

@pytest.fixture
def stability_qa4sm_file(TEST_DATA_DIR) -> Path:
    return Path(TEST_DATA_DIR / 'intra_annual' / 'stability' / '0-ESA_CCI_SM_passive.sm_with_1-ERA5_LAND.swvl1_tsw_stability_qa4sm.nc')

#------------------Helper functions------------------------


@log_function_call
def get_tmp_whole_test_data_dir(
        TEST_DATA_DIR: Path, tmp_paths: List[Path]) -> Tuple[Path, List[Path]]:
    '''Copy the whole test data directory to a temporary directory and return the path to the temporary directory

    Parameters
    ----------

    TEST_DATA_DIR: Path
        The path to the test data directory
    tmp_paths: List[Path]
        **Don't modify this list directly. Keeps track of created tmp dirs during a test run**

    Returns
    -------

    Tuple[Path, List[Path]]
        A tuple containing the path to the temporary directory and the list of temporary directories that have been created during the test
        '''
    if isinstance(TEST_DATA_DIR, str):
        TEST_DATA_DIR = Path(TEST_DATA_DIR)
    temp_dir = Path(tempfile.mkdtemp())
    shutil.copytree(TEST_DATA_DIR, temp_dir / TEST_DATA_DIR.name)

    return temp_dir / TEST_DATA_DIR.name, tmp_paths


@log_function_call
def get_tmp_single_test_file(test_file: Path,
                             tmp_paths: List[Path]) -> Tuple[Path, List[Path]]:
    '''Copy a single test file to a temporary directory and return the path to the temporary file

    Parameters
    ----------

    TEST_DATA_DIR: Path
        The path to the test data directory
    tmp_paths: List[Path]
        **Don't modify this list directly. Keeps track of created tmp files during a test run**

    Returns
    -------

    Tuple[Path, List[Path]]
        A tuple containing the path to the temporary file and the list of temporary files that have been created during the test
        '''
    if isinstance(test_file, str):
        test_file = Path(test_file)
    temp_dir = Path(tempfile.mkdtemp())
    temp_file_path = temp_dir / test_file.name
    shutil.copy(test_file, temp_file_path)
    return temp_file_path, tmp_paths


@log_function_call
def run_test_transcriber(
    ncfile: Path,
    intra_annual_slices: Union[None, TemporalSubWindowsCreator],
    keep_pytesmo_ncfile: bool,
    write_outfile: Optional[bool] = True
) -> Tuple[Pytesmo2Qa4smResultsTranscriber, xr.Dataset]:
    '''Run a test on the transcriber with the given parameters

    Parameters
    ----------

    ncfile: Path
        The path to the netcdf file to be transcribed
    intra_annual_slices: Union[None, TemporalSubWindowsCreator]
        The temporal sub-windows to be used for the transcription
    keep_pytesmo_ncfile: bool
        Whether to keep the original pytesmo nc file
    write_outfile: Optional[bool]
        Whether to write the transcribed dataset to a new netcdf file. Default is True

    Returns
    -------
    Tuple[Pytesmo2Qa4smResultsTranscriber, xr.Dataset]
        A tuple containing the transcriber instance and the transcribed dataset'''

    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=ncfile,
        intra_annual_slices=intra_annual_slices,
        keep_pytesmo_ncfile=keep_pytesmo_ncfile)

    logging.info(f"{transcriber=}")

    assert transcriber.exists

    transcriber.output_file_name = ncfile
    transcribed_ds = transcriber.get_transcribed_dataset()

    assert isinstance(transcribed_ds, xr.Dataset)

    if write_outfile:
        transcriber.write_to_netcdf(transcriber.output_file_name)
        assert Path(transcriber.output_file_name).exists()

    if keep_pytesmo_ncfile:
        assert Path(ncfile.parent,
                    ncfile.name + globals.OLD_NCFILE_SUFFIX).exists()
    else:
        assert not Path(ncfile.parent,
                        ncfile.name + globals.OLD_NCFILE_SUFFIX).exists()

    return transcriber, transcribed_ds


#------------------Check that all required consts are defined------------------
@log_function_call
def test_qr_globals_attributes():
    attributes = [
        'METRICS', 'TC_METRICS', 'NON_METRICS', 'METADATA_TEMPLATE',
        'IMPLEMENTED_COMPRESSIONS', 'ALLOWED_COMPRESSION_LEVELS',
        'INTRA_ANNUAL_METRIC_TEMPLATE', 'INTRA_ANNUAL_TCOL_METRIC_TEMPLATE',
        'TEMPORAL_SUB_WINDOW_SEPARATOR', 'DEFAULT_TSW',
        'TEMPORAL_SUB_WINDOW_NC_COORD_NAME', 'MAX_NUM_DS_PER_VAL_RUN',
        'DATASETS', 'TEMPORAL_SUB_WINDOWS'
    ]

    assert any(attr in dir(globals) for attr in attributes)

    assert 'zlib' in globals.IMPLEMENTED_COMPRESSIONS

    assert globals.ALLOWED_COMPRESSION_LEVELS == [None, *list(range(10))]

    assert globals.INTRA_ANNUAL_METRIC_TEMPLATE == [
        "{tsw}", globals.TEMPORAL_SUB_WINDOW_SEPARATOR, "{metric}"
    ]

    assert globals.INTRA_ANNUAL_TCOL_METRIC_TEMPLATE == globals.INTRA_ANNUAL_TCOL_METRIC_TEMPLATE == [
        "{tsw}", globals.TEMPORAL_SUB_WINDOW_SEPARATOR, "{metric}", "_",
        "{number}-{dataset}", "_between_"
    ]

    assert len(globals.TEMPORAL_SUB_WINDOW_SEPARATOR) == 1

    assert globals.TEMPORAL_SUB_WINDOWS == {
        "seasons": {
            "S1": [[12, 1], [2, 28]],
            "S2": [[3, 1], [5, 31]],
            "S3": [[6, 1], [8, 31]],
            "S4": [[9, 1], [11, 30]],
        },
        "months": {
            "Jan": [[1, 1], [1, 31]],
            "Feb": [[2, 1], [2, 28]],
            "Mar": [[3, 1], [3, 31]],
            "Apr": [[4, 1], [4, 30]],
            'May': [[5, 1], [5, 31]],
            "Jun": [[6, 1], [6, 30]],
            "Jul": [[7, 1], [7, 31]],
            "Aug": [[8, 1], [8, 31]],
            "Sep": [[9, 1], [9, 30]],
            "Oct": [[10, 1], [10, 31]],
            "Nov": [[11, 1], [11, 30]],
            "Dec": [[12, 1], [12, 31]],
        },
        "stability":{
        }
    }


# ------------------Test Pytesmo2Qa4smResultsTranscriber-------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

#------------Test instantiation of Pytesmo2Qa4smResultsTranscriber, attrs and basic functionalities------------


@log_function_call
def test_on_non_existing_file():
    with pytest.raises(FileNotFoundError):
        transcriber = Pytesmo2Qa4smResultsTranscriber(
            pytesmo_results='non_existing.nc',
            intra_annual_slices=None,
            keep_pytesmo_ncfile=False)


@log_function_call
def test_invalid_temp_subwins(seasonal_tsws_incl_bulk,
                              tmp_paths,
                              TEST_DATA_DIR,
                              test_file: Optional[Path] = None):
    logging.info(
        f'test_invalid_temp_subwins: {seasonal_tsws_incl_bulk=}, {tmp_paths=}, {TEST_DATA_DIR=}, {test_file=}'
    )
    if test_file is None:
        test_file = Path(TEST_DATA_DIR / 'basic' /
                         '0-ISMN.soil moisture_with_1-C3S.sm.nc')

    # Test that the transcriber raises an InvalidTemporalSubWindowError when the intra_annual_slices parameter is neither None nor a TemporalSubWindowsCreator instance
    tmp_test_file = get_tmp_single_test_file(test_file, tmp_paths)[0]
    with pytest.raises(InvalidTemporalSubWindowError):
        _, ds = run_test_transcriber(tmp_test_file,
                                     intra_annual_slices='faulty',
                                     keep_pytesmo_ncfile=False)
        ds.close()


@log_function_call
def test_invalid_temporalsubwindowscreator(seasonal_tsws_incl_bulk,
                                           tmp_paths,
                                           TEST_DATA_DIR,
                                           test_file: Optional[Path] = None):
    logging.info(
        f'test_invalid_temporalsubwindowscreator: {seasonal_tsws_incl_bulk=}, {tmp_paths=}, {TEST_DATA_DIR=}, {test_file=}'
    )
    if test_file is None:
        test_file = Path(TEST_DATA_DIR / 'basic' /
                         '0-ISMN.soil moisture_with_1-C3S.sm.nc')

    # Test that the transcriber raises an InvalidTemporalSubWindowError when the intra_annual_slices parameter is a faulty TemporalSubWindowsCreator instance
    tmp_test_file = get_tmp_single_test_file(test_file, tmp_paths)[0]

    with pytest.raises(InvalidTemporalSubWindowError):
        _, ds = run_test_transcriber(
            tmp_test_file,
            intra_annual_slices=TemporalSubWindowsCreator('gibberish'),
            keep_pytesmo_ncfile=False)
        ds.close()


@log_function_call
def test_temp_subwin_mismatch(seasonal_tsws_incl_bulk,
                              tmp_paths,
                              TEST_DATA_DIR,
                              test_file: Optional[Path] = None):
    logging.info(
        f'test_temp_subwin_mismatch: {seasonal_tsws_incl_bulk=}, {tmp_paths=}, {TEST_DATA_DIR=}, {test_file=}'
    )
    if test_file is None:
        test_file = Path(TEST_DATA_DIR / 'basic' /
                         '0-ISMN.soil moisture_with_1-C3S.sm.nc')

    # Test that the transcriber raises a TemporalSubWindowMismatchError when the intra_annual_slices parameter is a TemporalSubWindowsCreator instance that does not match the temporal sub-windows in the pytesmo_results file
    tmp_test_file = get_tmp_single_test_file(test_file, tmp_paths)[0]
    with pytest.raises(TemporalSubWindowMismatchError):
        _, ds = run_test_transcriber(
            tmp_test_file,
            intra_annual_slices=seasonal_tsws_incl_bulk,
            keep_pytesmo_ncfile=False)
        ds.close()


@log_function_call
def test_keep_pytesmo_ncfile(TEST_DATA_DIR, test_file: Optional[Path] = None):
    if test_file is None:
        test_file = Path(TEST_DATA_DIR / 'basic' /
                         '0-ISMN.soil moisture_with_1-C3S.sm.nc')
    tmp_test_file = get_tmp_single_test_file(test_file, tmp_paths)[0]
    transcriber, ds = run_test_transcriber(tmp_test_file,
                                           intra_annual_slices=None,
                                           keep_pytesmo_ncfile=True)
    transcriber.pytesmo_results.close()
    ds.close()


@log_function_call
def test_dont_keep_pytesmo_ncfile(TEST_DATA_DIR,
                                  test_file: Optional[Path] = None):
    if test_file is None:
        test_file = Path(TEST_DATA_DIR / 'basic' /
                         '0-ISMN.soil moisture_with_1-C3S.sm.nc')
    tmp_test_file = get_tmp_single_test_file(test_file, tmp_paths)[0]
    _, ds = run_test_transcriber(tmp_test_file,
                                 intra_annual_slices=None,
                                 keep_pytesmo_ncfile=False)
    ds.close()


@log_function_call
def test_ncfile_compression(TEST_DATA_DIR, test_file: Optional[Path] = None):
    if test_file is None:
        test_file = Path(TEST_DATA_DIR / 'basic' /
                         '0-ISMN.soil moisture_with_1-C3S.sm.nc')
    tmp_test_file = get_tmp_single_test_file(test_file, tmp_paths)[0]
    transcriber, ds = run_test_transcriber(tmp_test_file,
                                           intra_annual_slices=None,
                                           keep_pytesmo_ncfile=False,
                                           write_outfile=True)

    # only zlib compression is implemented so far, with compression levels 0-9
    with pytest.raises(NotImplementedError):
        transcriber.compress(transcriber.output_file_name, 'not_implemented',
                             0)
        transcriber.compress(transcriber.output_file_name, 'zlib', -1)
        transcriber.compress(transcriber.output_file_name, 'not_implemented',
                             -1)

    # test the case of a non-existing file
    assert not transcriber.compress('non_existing_file.nc', 'zlib', 0)

    # test successful compression with zlib and compression level 9
    assert transcriber.compress(transcriber.output_file_name, 'zlib', 9)

    # test successful compression with defaults
    assert transcriber.compress(transcriber.output_file_name)

    ds.close()


#-------------------Test default case (= no temporal sub-windows)--------------------------------------------

# todo: update this test accordnig to changes in the netcdf_transcription file - commented lines 526-551
@log_function_call
def test_bulk_case_transcription(TEST_DATA_DIR, tmp_paths):
    # Test transcription of all original test data nc files (== bulk case files)
    tmp_test_data_dir, _ = get_tmp_whole_test_data_dir(TEST_DATA_DIR,
                                                       tmp_paths)
    nc_files = [
        path for path in Path(tmp_test_data_dir).rglob('*.nc')
        if 'intra_annual' not in str(path)
    ]
    logging.info(f"Found {len(nc_files)} .nc files for transcription.")

    for i, ncf in enumerate(nc_files):
        _, ds = run_test_transcriber(ncf,
                                     intra_annual_slices=None,
                                     keep_pytesmo_ncfile=False,
                                     write_outfile=True)
        assert ds.sel(
            {globals.TEMPORAL_SUB_WINDOW_NC_COORD_NAME:
             globals.DEFAULT_TSW}) == globals.DEFAULT_TSW
        logging.info(f"Successfully transcribed file: {ncf}")
        ds.close()

    if tmp_test_data_dir.exists():
        shutil.rmtree(tmp_test_data_dir, ignore_errors=True)


#-------------------------------------------Test with intra-annual metrics---------------------------------------------


@log_function_call
def test_correct_file_transcription(seasonal_pytesmo_file, seasonal_qa4sm_file, monthly_pytesmo_file, monthly_qa4sm_file, stability_pytesmo_file, stability_qa4sm_file):
    '''
    Test the transcription of the test files with the correct temporal sub-windows and the correct output nc files'''

    # test that the test files exist
    assert seasonal_pytesmo_file.exists
    assert seasonal_qa4sm_file.exists
    assert monthly_pytesmo_file.exists
    assert monthly_qa4sm_file.exists
    assert stability_pytesmo_file.exists
    assert stability_qa4sm_file.exists

    # instantiate proper TemporalSubWindowsCreator instances for the corresponding test files
    bulk_tsw = NewSubWindow(
        'bulk', datetime(1900, 1, 1), datetime(2000, 1, 1)
    )  # if ever the default changes away from 'bulk, this will need to be taken into account

    seasons_tsws = TemporalSubWindowsCreator('seasons')
    seasons_tsws.add_temp_sub_wndw(bulk_tsw, insert_as_first_wndw=True)

    monthly_tsws = TemporalSubWindowsCreator('months')
    monthly_tsws.add_temp_sub_wndw(bulk_tsw, insert_as_first_wndw=True)

    stability_tsws = TemporalSubWindowsCreator(temporal_sub_window_type="stability")
    stability_tsws.add_temp_sub_wndw(bulk_tsw, insert_as_first_wndw=True)
        

    # Add annual sub-windows based on the years in the period
    period = [datetime(year=2009, month=1, day=1), datetime(year=2022, month=12, day=31)]

    stability_tsws = TemporalSubWindowsFactory._create_stability(0, period, None)

    # make sure the above defined temporal sub-windows are indeed the ones on the expected output nc files
    assert seasons_tsws.names == Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(seasonal_qa4sm_file)
    assert monthly_tsws.names == Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(monthly_qa4sm_file)
    assert stability_tsws.names == Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(stability_qa4sm_file)


    # instantiate transcribers for the test files
    seasonal_transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=seasonal_pytesmo_file,
        intra_annual_slices=seasons_tsws,
        keep_pytesmo_ncfile=False
    )  # deletion or keeping of the original pytesmo nc file only triggers when the transcriber is written to a new file, which is not the case here

    monthly_transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=monthly_pytesmo_file,
        intra_annual_slices=monthly_tsws,
        keep_pytesmo_ncfile=False)

    stability_transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=stability_pytesmo_file,
        intra_annual_slices=stability_tsws,
        keep_pytesmo_ncfile=False)

    assert seasonal_transcriber.exists
    assert monthly_transcriber.exists
    assert stability_transcriber.exists

    # get the transcribed datasets
    seasonal_transcribed_ds = seasonal_transcriber.get_transcribed_dataset()
    monthly_transcribed_ds = monthly_transcriber.get_transcribed_dataset()
    stability_transcribed_ds = stability_transcriber.get_transcribed_dataset()

    # check that the transcribed datasets are indeed xarray.Dataset instances
    assert isinstance(seasonal_transcribed_ds, xr.Dataset)
    assert isinstance(monthly_transcribed_ds, xr.Dataset)
    assert isinstance(stability_transcribed_ds, xr.Dataset)

    # check that the transcribed datasets are equal to the expected output files
    # xr.testing.assert_equal(ds1, ds2) runs a more detailed comparison of the two datasets as compared to ds1.equals(ds2)
    with xr.open_dataset(seasonal_qa4sm_file) as f:
        expected_seasonal_ds = f
    with xr.open_dataset(monthly_qa4sm_file) as f:
        expected_monthly_ds = f
    with xr.open_dataset(stability_qa4sm_file) as f:
        expected_stability_ds = f

    #!NOTE: pytesmo/QA4SM offer the possibility to calculate Kendall's tau, but currently this metric is deactivated.
    #!      Therefore, in a real validation run no tau related metrics will be transcribed to the QA4SM file, even though they might be present in the pytesmo file.

    # drop the tau related metrics from the expected datasets
    for var in expected_seasonal_ds.data_vars:
        if 'tau' in var:
            logging.info(
                f"Dropping variable {var} from expected seasonal dataset")
            expected_seasonal_ds = expected_seasonal_ds.drop_vars(var)

    for var in expected_monthly_ds.data_vars:
        if 'tau' in var:
            logging.info(
                f"Dropping variable {var} from expected monthly dataset")
            expected_monthly_ds = expected_monthly_ds.drop_vars(var)

    # returns None if the datasets are equal
    assert xr.testing.assert_equal(
        monthly_transcribed_ds,
        expected_monthly_ds) is None
    # returns None if the datasets are equal
    assert xr.testing.assert_equal(
        seasonal_transcribed_ds,
        expected_seasonal_ds) is None
    # returns None if the datasets are equal
    assert xr.testing.assert_equal(stability_transcribed_ds,
                                   expected_stability_ds) is None

    # the method above does not check attrs of the datasets, so we do it here
    # Creation date and qa4sm_reader might differ, so we exclude them from the comparison
    datasets = [
        monthly_transcribed_ds, expected_monthly_ds, seasonal_transcribed_ds,
        expected_seasonal_ds
    , stability_transcribed_ds, expected_stability_ds]
    attrs_to_be_excluded = ['date_created', 'qa4sm_version']
    for ds in datasets:
        for attr in attrs_to_be_excluded:
            if attr in ds.attrs:
                del ds.attrs[attr]

    assert seasonal_transcribed_ds.attrs == expected_seasonal_ds.attrs
    assert monthly_transcribed_ds.attrs == expected_monthly_ds.attrs
    assert stability_transcribed_ds.attrs == expected_stability_ds.attrs

    # Compare the coordinate attributes
    for coord in seasonal_transcribed_ds.coords:
        for attr in seasonal_transcribed_ds[coord].attrs:
            if isinstance(seasonal_transcribed_ds[coord].attrs[attr],
                          (list, np.ndarray)):
                assert np.array_equal(
                    seasonal_transcribed_ds[coord].attrs[attr],
                    expected_seasonal_ds[coord].attrs[attr]
                ), f"Attributes for coordinate {coord} do not match in seasonal dataset"
            else:
                assert seasonal_transcribed_ds[coord].attrs[
                    attr] == expected_seasonal_ds[coord].attrs[
                        attr], f"Attributes for coordinate {coord} do not match in seasonal dataset: '{seasonal_transcribed_ds[coord].attrs[attr]}' =! '{expected_seasonal_ds[coord].attrs[attr]}'"

    for coord in monthly_transcribed_ds.coords:
        for attr in monthly_transcribed_ds[coord].attrs:
            if isinstance(monthly_transcribed_ds[coord].attrs[attr],
                          (list, np.ndarray)):
                assert np.array_equal(
                    monthly_transcribed_ds[coord].attrs[attr],
                    expected_monthly_ds[coord].attrs[attr]
                ), f"Attributes for coordinate {coord} do not match in monthly dataset"
            else:
                assert monthly_transcribed_ds[coord].attrs[
                    attr] == expected_monthly_ds[coord].attrs[
                        attr], f"Attributes for coordinate {coord} do not match in monthly dataset: '{monthly_transcribed_ds[coord].attrs[attr]}' =! '{expected_monthly_ds[coord].attrs[attr]}'"

    seasonal_transcribed_ds.close()
    monthly_transcribed_ds.close()
    stability_transcribed_ds.close()


#TODO: refactoring
@log_function_call
def test_plotting(seasonal_qa4sm_file, monthly_qa4sm_file, stability_qa4sm_file, tmp_paths):
    '''
    Test the plotting of the test files with temporal sub-windows beyond the bulk case (this scenario covered in other tests)
    '''

    tmp_seasonal_file, _ = get_tmp_single_test_file(seasonal_qa4sm_file,
                                                    tmp_paths)
    tmp_seasonal_dir = tmp_seasonal_file.parent

    tmp_monthly_file, _ = get_tmp_single_test_file(monthly_qa4sm_file,
                                                   tmp_paths)
    tmp_monthly_dir = tmp_monthly_file.parent

    tmp_stability_file, _ = get_tmp_single_test_file(stability_qa4sm_file, tmp_paths)
    
    tmp_stability_dir = tmp_stability_file.parent
    
    # check the output directories

    pa.plot_all(
        filepath=tmp_seasonal_file,
        temporal_sub_windows=Pytesmo2Qa4smResultsTranscriber.
        get_tsws_from_ncfile(tmp_seasonal_file),
        out_dir=tmp_seasonal_dir,
        save_all=True,
        out_type=['png', 'svg'],
    )

    metrics_not_plotted = [*globals.metric_groups['common'], *globals.metric_groups['triple'], *globals._metadata_exclude]

    tsw_dirs_expected = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(
        tmp_seasonal_file)
    if globals.DEFAULT_TSW in tsw_dirs_expected:
        tsw_dirs_expected.remove(
            globals.DEFAULT_TSW)  # we're not checking the default case here

    for tsw in tsw_dirs_expected:
        assert Path(
            tmp_seasonal_dir /
            tsw).is_dir(), f"{tmp_seasonal_dir / tsw} is not a directory"

        # only metrics and tcol metrics get their dedicated plots for each temporal sub-window
        for metric in [
                *list(globals.METRICS.keys()), *list(globals.TC_METRICS.keys())
        ]:
            if metric in metrics_not_plotted:
                continue
            assert Path(
                tmp_seasonal_dir / tsw / f"{tsw}_boxplot_{metric}.png"
            ).exists(
            ), f"{tmp_seasonal_dir / tsw / f'{tsw}_boxplot_{metric}.png'} does not exist"
            assert Path(
                tmp_seasonal_dir / tsw / f"{tsw}_boxplot_{metric}.svg"
            ).exists(
            ), f"{tmp_seasonal_dir / tsw / f'{tsw}_boxplot_{metric}.svg'} does not exist"

        assert Path(
            tmp_seasonal_dir / tsw / f'{tsw}_statistics_table.csv'
        ).is_file(
        ), f"{tmp_seasonal_dir / tsw / f'{tsw}_statistics_table.csv'} does not exist"

    # check intra-annual-metric-exclusive comparison boxplots
    assert Path(tmp_seasonal_dir / 'comparison_boxplots').is_dir()
    for metric in globals.METRICS:
        if metric in metrics_not_plotted:
            continue
        assert Path(
            tmp_seasonal_dir / 'comparison_boxplots' /
            globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric,
                                                       filetype='png')
        ).exists(
        ), f"{tmp_seasonal_dir / 'comparison_boxplots' / globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric, filetype='png')} does not exist"
        assert Path(
            tmp_seasonal_dir / 'comparison_boxplots' /
            globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric,
                                                       filetype='svg')
        ).exists(
        ), f"{tmp_seasonal_dir / 'comparison_boxplots' / globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric, filetype='svg')} does not exist"

    # now check the file with monthly temporal sub-windows and without tcol metrics

    pa.plot_all(
        filepath=tmp_monthly_file,
        temporal_sub_windows=Pytesmo2Qa4smResultsTranscriber.
        get_tsws_from_ncfile(tmp_monthly_file),
        out_dir=tmp_monthly_dir,
        save_all=True,
        save_metadata=True,
        out_type=['png', 'svg'],
    )

    tsw_dirs_expected = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(
        tmp_monthly_file)
    if globals.DEFAULT_TSW in tsw_dirs_expected:
        tsw_dirs_expected.remove(globals.DEFAULT_TSW)

    for t, tsw in enumerate(tsw_dirs_expected):
        assert Path(
            tmp_monthly_dir /
            tsw).is_dir(), f"{tmp_monthly_dir / tsw} is not a directory"

        # no tcol metrics present here
        for metric in [*list(globals.METRICS.keys())]:
            if metric in metrics_not_plotted:
                continue
            # tsw specific plots
            assert Path(
                tmp_monthly_dir / tsw / f"{tsw}_boxplot_{metric}.png"
            ).exists(
            ), f"{tmp_monthly_dir / tsw / f'{tsw}_boxplot_{metric}.png'} does not exist"
            assert Path(
                tmp_monthly_dir / tsw / f"{tsw}_boxplot_{metric}.svg"
            ).exists(
            ), f"{tmp_monthly_dir / tsw / f'{tsw}_boxplot_{metric}.svg'} does not exist"

            if t == 0:
                #comparison boxplots
                assert Path(tmp_seasonal_dir / 'comparison_boxplots').is_dir()
                assert Path(
                    tmp_seasonal_dir / 'comparison_boxplots' /
                    globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric,
                                                               filetype='png')
                ).exists(
                ), f"{tmp_seasonal_dir / 'comparison_boxplots' / globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric, filetype='png')} does not exist"
                assert Path(
                    tmp_seasonal_dir / 'comparison_boxplots' /
                    globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric,
                                                               filetype='svg')
                ).exists(
                ), f"{tmp_seasonal_dir / 'comparison_boxplots' / globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric, filetype='svg')} does not exist"
        assert Path(
            tmp_monthly_dir / tsw / f'{tsw}_statistics_table.csv'
        ).is_file(
        ), f"{tmp_monthly_dir / tsw / f'{tsw}_statistics_table.csv'} does not exist"


    # now check the file with stability temporal sub-windows and without tcol metrics and the count of the plots

    pa.plot_all(
        filepath=tmp_stability_file,
        temporal_sub_windows=Pytesmo2Qa4smResultsTranscriber.
        get_tsws_from_ncfile(tmp_stability_file),
        out_dir=tmp_stability_dir,
        save_all=True,
        save_metadata=True,
        out_type=['png', 'svg'],
    )

    tsw_dirs_expected = Pytesmo2Qa4smResultsTranscriber.get_tsws_from_ncfile(
        tmp_stability_file)

    # Subfolders for tsw should not exist in the stability - case
    for t, tsw in enumerate(tsw_dirs_expected):
        if tsw == globals.DEFAULT_TSW:
            assert Path(tmp_stability_dir / tsw).is_dir(), f"{tmp_stability_dir / tsw} is not a directory"
        else:
            assert not Path(tmp_stability_dir / tsw).exists(), f"{tmp_stability_dir / tsw} should not exist"
            continue

        # no tcol metrics present here
        for metric in [*list(globals.METRICS.keys())]:
            if metric in metrics_not_plotted:
                continue
            # tsw specific plots
            assert Path(
                tmp_stability_dir / tsw / f"{tsw}_boxplot_{metric}.png"
            ).exists(
            ), f"{tmp_stability_dir / tsw / f'{tsw}_boxplot_{metric}.png'} does not exist"
            assert Path(
                tmp_stability_dir / tsw / f"{tsw}_boxplot_{metric}.svg"
            ).exists(
            ), f"{tmp_stability_dir / tsw / f'{tsw}_boxplot_{metric}.svg'} does not exist"

            if t == 0:
                #comparison boxplots
                assert Path(tmp_stability_dir / 'comparison_boxplots').is_dir()
                assert Path(
                    tmp_stability_dir / 'comparison_boxplots' /
                    globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric,
                                                                filetype='png')
                ).exists(
                ), f"{tmp_stability_dir / 'comparison_boxplots' / globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric, filetype='png')} does not exist"
                assert Path(
                    tmp_stability_dir / 'comparison_boxplots' /
                    globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric,
                                                                filetype='svg')
                ).exists(
                ), f"{tmp_stability_dir / 'comparison_boxplots' / globals.CLUSTERED_BOX_PLOT_SAVENAME.format(metric=metric, filetype='svg')} does not exist"
        assert Path(
            tmp_stability_dir / tsw / f'{tsw}_statistics_table.csv'
        ).is_file(
        ), f"{tmp_stability_dir / tsw / f'{tsw}_statistics_table.csv'} does not exist"
    
        plot_dir = Path(tmp_stability_dir / globals.DEFAULT_TSW)
        assert len(list(plot_dir.iterdir())) == 69
        assert all(file.suffix in [".png", ".svg", ".csv"] for file in plot_dir.iterdir()), "Not all files have been saved as .png or .csv"


@log_function_call
def test_write_to_netcdf_default(TEST_DATA_DIR, tmp_paths):
    temp_netcdf_file: Path = get_tmp_single_test_file(
        Path(TEST_DATA_DIR / 'basic' /
             '0-ISMN.soil moisture_with_1-C3S.sm.nc'), tmp_paths)[0]
    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=temp_netcdf_file)

    transcribed_ds = transcriber.get_transcribed_dataset()
    # Write to NetCDF
    transcriber.write_to_netcdf(temp_netcdf_file)

    # Check if the file is created
    assert temp_netcdf_file.exists()

    # Close the datasets
    transcriber.pytesmo_results.close()
    transcriber.transcribed_dataset.close()
    transcribed_ds.close()


@log_function_call
def test_write_to_netcdf_custom_encoding(TEST_DATA_DIR, tmp_paths):
    temp_netcdf_file: Path = get_tmp_single_test_file(
        Path(TEST_DATA_DIR / 'basic' /
             '0-ISMN.soil moisture_with_1-C3S.sm.nc'), tmp_paths)[0]
    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=temp_netcdf_file)

    transcribed_ds = transcriber.get_transcribed_dataset()

    custom_encoding = {
        str(var): {
            'zlib': True,
            'complevel': 1
        }
        for var in transcribed_ds.variables
        if not np.issubdtype(transcribed_ds[var].dtype, np.object_)
    }

    # Write to NetCDF with custom encoding
    transcriber.write_to_netcdf(temp_netcdf_file, encoding=custom_encoding)

    # Check if the file is created
    assert temp_netcdf_file.exists()

    # Close the datasets
    transcriber.pytesmo_results.close()
    transcriber.transcribed_dataset.close()
    transcribed_ds.close()


def test_get_transcribed_dataset(TEST_DATA_DIR, tmp_paths):
    temp_netcdf_file = get_tmp_single_test_file(
        Path(TEST_DATA_DIR / 'basic' /
             '0-ISMN.soil moisture_with_1-C3S.sm.nc'), tmp_paths)[0]
    transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=temp_netcdf_file)

    # Get the transcribed dataset
    transcribed_dataset = transcriber.get_transcribed_dataset()

    # Check if the transcribed dataset is an xarray Dataset
    assert isinstance(transcribed_dataset, xr.Dataset)

    # Close the datasets
    transcriber.pytesmo_results.close()
    transcriber.transcribed_dataset.close()
    transcribed_dataset.close()


@log_function_call
def test_is_valid_metric_name(seasonal_pytesmo_file, seasonal_tsws_incl_bulk):
    # Create a mock cases
    mock_tsws = seasonal_tsws_incl_bulk
    mock_transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=seasonal_pytesmo_file,
        intra_annual_slices=mock_tsws,
        keep_pytesmo_ncfile=False)

    # Test valid metric names
    tsws = mock_tsws.names
    sep = '|'
    dataset_combi = '_between_0-ERA5_and_3-ESA_CCI_SM_combined'
    valid_metrics = globals.METRICS.keys()

    valid_metric_names = [
        f'{tsw}{sep}{metric}{dataset_combi}' for tsw in tsws
        for metric in valid_metrics
    ]
    for metric_name in valid_metric_names:
        assert mock_transcriber.is_valid_metric_name(metric_name) == True

    # Test invalid metric names with metrics that dont even exist
    nonsense_metrics = ['nonsense_metric_1', 'nonsense_metric_2']
    nonsense_metric_names = [
        f'{tsw}{sep}{metric}{dataset_combi}' for tsw in tsws
        for metric in nonsense_metrics
    ]
    for metric_name in nonsense_metric_names:
        assert mock_transcriber.is_valid_metric_name(metric_name) == False

    # Test tcol metric names
    tcol_metrics = globals.TC_METRICS.keys()
    tcol_metric_names = [
        f'{tsw}{sep}{metric}{dataset_combi}' for tsw in tsws
        for metric in tcol_metrics
    ]
    for metric_name in tcol_metric_names:
        assert mock_transcriber.is_valid_metric_name(metric_name) == False


@log_function_call
def test_is_valid_tcol_metric_name(seasonal_pytesmo_file,
                                   seasonal_tsws_incl_bulk):
    # Create a mock cases
    mock_tsws = seasonal_tsws_incl_bulk
    mock_transcriber = Pytesmo2Qa4smResultsTranscriber(
        pytesmo_results=seasonal_pytesmo_file,
        intra_annual_slices=mock_tsws,
        keep_pytesmo_ncfile=False)

    tcol_metric_names = [
        'S1|snr_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|snr_2-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|snr_3-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_3-ESA_CCI_SM_combined',
        'S1|snr_4-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_4-ESA_CCI_SM_combined',
    ]  #amongst others

    for metric_name in tcol_metric_names:
        assert mock_transcriber.is_valid_tcol_metric_name(metric_name) == True

    tcol_metrics_not_transcribed = [
        'S1|snr_ci_lower_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|snr_ci_lower_2-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|err_std_ci_lower_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|err_std_ci_lower_2-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|beta_ci_lower_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|beta_ci_lower_2-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_2-ESA_CCI_SM_combined',
        'S1|snr_ci_lower_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_3-ESA_CCI_SM_combined',
        'S1|snr_ci_lower_3-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_3-ESA_CCI_SM_combined',
        'S1|err_std_ci_lower_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_3-ESA_CCI_SM_combined',
        'S1|err_std_ci_lower_3-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_3-ESA_CCI_SM_combined',
        'S1|beta_ci_lower_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_3-ESA_CCI_SM_combined',
        'S1|beta_ci_lower_3-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_3-ESA_CCI_SM_combined',
        'S1|snr_ci_lower_1-ESA_CCI_SM_combined_between_0-ERA5_and_1-ESA_CCI_SM_combined_and_4-ESA_CCI_SM_combined',
    ]

    for metric_name in tcol_metrics_not_transcribed:
        assert mock_transcriber.is_valid_tcol_metric_name(metric_name) == False


if __name__ == '__main__':
    test_file = Path('/tmp/test_dir/0-ISMN.soil_moisture_with_1-C3S.sm.nc')
    # transcriber, ds = run_test_transcriber(test_file,
    #                                        intra_annual_slices=None,
    #                                        keep_pytesmo_ncfile=True)
    # transcriber.pytesmo_results.close()
    # ds.close()
    test_bulk_case_transcription()