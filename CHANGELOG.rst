=========
Changelog
=========

Unreleased changes in master branch
===================================

Version 0.12.1
==============
- Bugfix comparison module boxplot creation

Version 0.12
============
- Standardized plots with unified sizing, fonts, labels, and colorblind-safe palettes
- Streamlined titles by removing repetitive text and relocating supplementary details
- Fixed dataset combinations to correctly display all possible pairs
- Refined UI elements including logos, text wrapping, and legend designs
- Pin netcdf4 to <1.7.3

Version 0.11.6
==============
- Bugfix creation of comparison mapplot

Version 0.11.5
==============
- Update build.yml
- Fix comparison module
- Results with val_is_scattered_data == 'True' rendered as point data in plots

Version 0.11.4
==============
- Added Custom Plot Functions for Users support materials

Version 0.11.3
==============
- Added fallback names for new CCI and C3S data
- Pyscaffold version updated, use pyproject.toml instead of setup.cfg

Version 0.11.2
==============
- Update dependencies, remove environment.yml
- Fix empty comparison plot generation in stability case

Version 0.11.1
==============
- ISMN metadata boxplots split by network (`PR #97 <https://github.com/awst-austria/qa4sm-reader/pull/97>`_)
- Fix filenames of triple collocation mapplots
- Restrict qtbase to versions <= 6.7.2

Version 0.11
============
- Environment updated to support Python 3.12
- numpy updated to >2, and cartopy updated to 0.22.0
- libtiff version pinned to 4.5.1 due to compatibility issues with Ubuntu 24
- GitHub Actions updated to run tests for new Python versions
- geos version pinned for stability
- Updated setup file
- Moved cartopy dependency to PyPI

Version 0.10
============
- Fix bugs with metadata plots by @wpreimes in https://github.com/awst-austria/qa4sm-reader/pull/78
- Update environment.yml by @sheenaze in https://github.com/awst-austria/qa4sm-reader/pull/81
- fixed failing windows miniconda setup by @nfb2021 in https://github.com/awst-austria/qa4sm-reader/pull/82
- Required update to handle new QA4SM netCDF structure according to Jir… by @wpreimes in https://github.com/awst-austria/qa4sm-reader/pull/83
- Fix comparison plot creation logic by @nfb2021 in https://github.com/awst-austria/qa4sm-reader/pull/87
- Add Stability Metrics Plots by @daberer in https://github.com/awst-austria/qa4sm-reader/pull/86
- Bugfix labels for slopeR plots by @daberer in https://github.com/awst-austria/qa4sm-reader/pull/88
- Bugfix temporal sub-windows period by @daberer in https://github.com/awst-austria/qa4sm-reader/pull/89

Version 0.9.1
=============
- Added function to retrieve the package version

Version 0.9
===========
- QA4SM release 2 version
- Status maps and box plots
- FRM box plots added
- Scaling reference units used on plots
- SMOS and SMAP L2 added

Version 0.8
===========
- Reader adaptation for plotting validation error maps and barplots.
- Based on the "status_" keyword variable in the output netCDF files.

Version 0.7.5
=============
- The comparison plots for TCA metrics were shown with inverted values. The issue is now fixed.

Version 0.7.4
=============
- New formula added for the averaging of non-additive scores (Pearson's and Spearman's correlation)
- Improved layout in comparison plots
- Code formatting established (through yapf and pep8)

Version 0.7.3
=============
- Fix bug in metadata plots download; improves some graphics

Version 0.7.2
=============
- Modified to generate mapplots from SMOS L3 as reference. Includes small change in metadata plots labels.

Version 0.7.1
=============
- Add alternative dataset names

Version 0.7
===========
- Added a new dataset to the reader (SMOS L3), including functions to deal with missing datasets specifications (version name, units, ..)

Version 0.6.4
=============
- Added a new dataset to the reader (SMOS L3), including functions to deal with missing datasets specifications (version name, units, ..)

Version 0.6.3
=============
- Updated project requirements.txt file to overcome VersionConflict exception in pypi

Version 0.6.2
=============
- Methods have been implemented to generate a .csv file including the statistics summary table at the end of the validation.

Version 0.6.1
=============
- Small release to fix compatibility with QA4SM and some code cleanup

Version 0.6.0
=============
- Includes methods to read the metadata variables from the output netCDF file. The plot_all function in qa4sm_reader.plot_all.py has now a switch that produces (if the necessary information is available in the output file) metadata boxplots based on:
 - Land cover classes (CCI Landcover)
 - Climate classes (Koeppen-Geiger classification)
 - Soil granulometry (coarse - medium - fine)
- All the tests have been updated to accomodate for this.

Version 0.5.2
=============
- The environment of the reader was updated to cartopy==0.20.0 to solve issues with broken urls in the previous Cartopy version. The continuous integration tests were consequently updated to span on versions 3.7 to 3.9 of python - successfully

Version 0.5
===========
- The new comparison.py module has been added with relative tests
- A notebook has been included to show the usage of the comparison modul

Version 0.4
===========
- update on plots of datasets with irregular grids
- Quick inspection table added
- IQC instead of StdDev added to plots

Version 0.3.4
=============
- Switch from Travis CI to GitHub Actions
- Allow plotting from irregular grids (SMOS, ASCAT)

Version 0.3.3
=============
- Fix bug that lead to failing plots for CCI combined

Version 0.3.2
=============
- Fix bug that could break global overview maps
- Resolve deprecation warnings caused by cartopy

Version 0.3.1
=============
- Change how plots are named
 
Version 0.3
===========
- Fixes for integration in QA4SM Prod

Version 0.2
===========
- Updates for TC, refactoring

Version 0.1
===========
- First implementation


