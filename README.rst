#############
``guptri_py``
#############

.. image:: https://readthedocs.org/projects/guptri-py/badge/?version=latest
   :target: https://guptri-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

***************************************
A GUPTRI wrapper for NumPy and SageMath
***************************************

This Python package provides Python bindings for the software GUPTRI_ and
can be used with `NumPy <NUMPY_>`_ and, optionally, `SageMath <SAGE_>`_.

GUPTRI is a Fortran library by Jim Demmel and Bo Kågström for robust
computation of generalized eigenvalues of singular matrix pencils.
Standard tools like LAPACK do not reliably handle singular generalized
eigenvalue problems.

GUPTRI solves this by computing a generalized block upper triangular form
(generalized Schur staircase form) of a matrix pencil, revealing the Kronecker
structure of the pencil. For details, see the `documentation <guptri_py_rtd_>`_
and the references therein.

Examples
========

See the examples and documentation at
`https://guptri-py.readthedocs.io <guptri_py_rtd_>`_.

Installation
============

**Requirements**: `NumPy <NUMPY_>`_ and, optionally, `SageMath <SAGE_>`_
(tested with Sage 9.6 on Arch Linux and with earlier versions on macOS).

First, clone the `repository from GitHub <guptri_py_gh_>`_::

    git clone https://github.com/mwageringel/guptri_py.git && cd guptri_py

To install with Python 3 and NumPy, run the following command::

    pip3 install --upgrade --no-index -v .

Alternatively, for use with Sage, run this command::

    sage -pip install --upgrade --no-index -v .

To install into the Python user install directory (no root access required),
use::

    sage -pip install --upgrade --no-index -v --user .

After successful installation, run the tests with Sage::

    sage -t guptri_py

Installing into a virtual environment (with system packages)::

    python -m venv --system-site-packages ./venv   # assumes that setuptools and numpy are installed system-wide

    ./venv/bin/pip3 install --upgrade --no-index -v .
    # ./venv/bin/pip3 install --upgrade --no-index -v --no-build-isolation .

    cd ./venv && ./bin/python   # it is important to change to a different directory
    cd ./venv && ./bin/python -m IPython   # (or using IPython)

Installing into a virtual environment (without system packages)::

    python -m venv ./venv
    ./venv/bin/pip3 install numpy
    ./venv/bin/pip3 install wheel
    ./venv/bin/pip3 install --upgrade --no-index -v .
    cd ./venv && ./bin/python -m IPython   # (or using IPython)

Issues
------

* With Python 3.12+ and NumPy 1.26+, the build currently fails as numpy.distutils has been removed (#1).
* With NumPy ≤ 1.17, it may be necessary to set::

    export NPY_DISTUTILS_APPEND_FLAGS=1

  to fix a linking problem. See https://github.com/numpy/numpy/issues/12799.

.. _SAGE: https://www.sagemath.org/
.. _GUPTRI: https://web.archive.org/web/20080920172251/https://www8.cs.umu.se/research/nla/singular_pairs/guptri/
.. _NUMPY: https://numpy.org/
.. _guptri_py_gh: https://github.com/mwageringel/guptri_py
.. _guptri_py_rtd: https://guptri-py.readthedocs.io/en/latest/#module-guptri_py
