============
qa4sm_reader
============

|ci| |cov| |pip|

.. |ci| image:: https://github.com/awst-austria/qa4sm-reader/actions/workflows/build.yml/badge.svg?branch=master
   :target: https://github.com/awst-austria/qa4sm-reader/actions

.. |cov| image:: https://coveralls.io/repos/awst-austria/qa4sm-reader/badge.png?branch=master
  :target: https://coveralls.io/r/awst-austria/qa4sm-reader?branch=master

.. |pip| image:: https://badge.fury.io/py/qa4sm-reader.svg
    :target: https://badge.fury.io/py/qa4sm-reader


qa4sm_reader is a python package to read and plot the result files of the `qa4sm service`_.


Installation
============

This package should be installable through pip

.. code::

    pip install qa4sm_reader

Usage
=====

This package is used to analyze a qa4sm netCDF output file and produce all relevant plots and maps.

Development Setup
=================

The project was setup using `pyscaffold`_ and closely follows the recommendations.

Install Dependencies
--------------------

For Development we recommend creating a ``conda`` environment.

.. code::

    cd qa4sm-reader
    conda env create python=3.10 #  create environment from requirements.rst
    conda activate qa4sm_reader
    conda env update -f environment.yml -n qa4sm_reader
    pip install -e .

To remove the environment again, run:

.. code::

    conda deactivate
    conda env remove -n qa4sm_reader

Code Formatting
---------------
To apply pep8 conform styling to any changed files [we use `yapf`](https://github.com/google/yapf). The correct
settings are already set in `setup.cfg`. Therefore the following command
should be enough:

.. code::

    yapf file.py --in-place

Testing
-------

For testing, we use ``py.test``:

.. code::

    pytest


The dependencies are automatically installed by `pytest-runner`_ when you run the tests. The test-dependencies are listed in the ``testing`` field inside the ``[options.extras_require]`` section of ``setup.cfg``.
For some reasons, the dependencies are not installed as expected. To workaround, do:

.. code::

    pip install pytest-cov

The files used for testing are included in this package. They are however subject to other `terms and conditions`_.

Known Issues
------------

No known issues - please `open an issue`_ in case you come across a malfunctioning in the package.


.. _qa4sm service: https://qa4sm.eu
.. _pyscaffold: https://pyscaffold.org
.. _pytest-runner: https://github.com/pytest-dev/pytest-runner
.. _terms and conditions: https://qa4sm.eu/terms
.. _open an issue: https://github.com/awst-austria/qa4sm-reader/issues
