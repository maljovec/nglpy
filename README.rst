=====
nglpy
=====

.. badges

.. image:: https://img.shields.io/pypi/v/nglpy.svg
        :target: https://pypi.python.org/pypi/nglpy
        :alt: Latest Version on PyPI
.. image:: https://img.shields.io/pypi/dm/nglpy.svg?label=PyPI%20downloads
        :target: https://pypi.org/project/nglpy/
        :alt: PyPI downloads

.. image:: https://github.com/maljovec/nglpy/actions/workflows/quality.yaml/badge.svg?branch=main
        :target: https://github.com/maljovec/nglpy/actions
        :alt: Code Quality Test Results
.. image:: https://github.com/maljovec/nglpy/actions/workflows/test.yaml/badge.svg?branch=main
        :target: https://github.com/maljovec/nglpy/actions
        :alt: Test Suite Results

.. image:: https://www.codefactor.io/repository/github/maljovec/nglpy/badge
        :target: https://www.codefactor.io/repository/github/maljovec/nglpy
        :alt: CodeFactor
.. image:: https://coveralls.io/repos/github/maljovec/nglpy/badge.svg?branch=main
        :target: https://coveralls.io/github/maljovec/nglpy?branch=main
        :alt: Coveralls
.. image:: https://readthedocs.org/projects/nglpy/badge/?version=latest
        :target: https://nglpy.readthedocs.io/en/latest/?badge=latest
        :alt: ReadTheDocs
.. image:: https://pyup.io/repos/github/maljovec/nglpy/shield.svg
        :target: https://pyup.io/repos/github/maljovec/nglpy/
        :alt: Pyup

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
        :target: https://github.com/psf/black
        :alt: This code is formatted in black
.. image:: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
        :target: https://opensource.org/licenses/BSD-3-Clause
        :alt: BSD 3-Clause License

.. end_badges

.. logo

.. image:: docs/_static/nglpy.svg
    :align: center
    :alt: nglpy

.. end_logo

.. introduction

A Python wrapped version of the [Neighborhood Graph Library
(NGL_) developed by Carlos Correa and Peter Lindstrom.

.. _NGL: http://www.ngraph.org/

.. LONG_DESCRIPTION

Given a set of arbitrarily arranged points in any dimension, this library is
able to construct several different types of neighborhood graphs mainly focusing
on empty region graph algorithms such as the beta skeleton family of graphs.

.. END_LONG_DESCRIPTION

.. end_introduction

.. install

Installation
============

::

    pip install nglpy

.. end-install

.. usage

Usage
=====

Then you can use the library from python such as the example below::

    import nglpy
    import numpy as np

    point_set = np.random.rand(100,2)
    max_neighbors = 9
    beta = 1

    aGraph = nglpy.EmptyRegionGraph(max_neighbors=max_neighbors, relaxed=False, beta=beta)
    aGraph.build(point_set)

    aGraph.neighbors()

.. end-usage
