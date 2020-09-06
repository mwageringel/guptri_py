#############
``guptri_py``
#############

.. image:: https://travis-ci.com/mwageringel/guptri_py.svg?branch=master
   :target: https://travis-ci.com/mwageringel/guptri_py
   :alt: Build Status
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
(tested with Ubuntu 20.04 and macOS, Sage 9.1, and on Travis-CI with Ubuntu).

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

Issues
------

* With NumPy ≤ 1.17, it may be necessary to set::

    export NPY_DISTUTILS_APPEND_FLAGS=1

  to fix a linking problem. See https://github.com/numpy/numpy/issues/12799.

.. _SAGE: https://www.sagemath.org/
.. _GUPTRI: https://www8.cs.umu.se/research/nla/singular_pairs/guptri/
.. _NUMPY: https://numpy.org/
.. _guptri_py_gh: https://github.com/mwageringel/guptri_py
.. _guptri_py_rtd: https://guptri-py.readthedocs.io/en/latest/#module-guptri_py
