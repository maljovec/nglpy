# nglpy
[![Build Status](https://travis-ci.org/maljovec/nglpy.svg?branch=master)](https://travis-ci.org/maljovec/nglpy)

A Python wrapped version of the [Neighborhood Graph Library
(NGL)](http://www.ngraph.org/) developed by Carlos Correa and Peter Lindstrom.

[//]: # (LONG_DESCRIPTION)

Given a set of arbitrarily arranged points in any dimension, this library is
able to construct several different types of neighborhood graphs mainly focusing
on empty region graph algorithms such as the beta skeleton family of graphs.

[//]: # (END_LONG_DESCRIPTION)

# Installation

(Currently under test in test.pypi.org)

```
pip install nglpy
```

Then you can use the library from python such as the example below:

```python
import nglpy
import numpy as np

point_set = np.random.rand(100,2)
max_neighbors = 9
beta = 1

## TODO: Make this an enum, remove hard-coding
graph_type = 'beta skeleton'

aGraph = nglpy.Graph(point_set, graph_type, max_neighbors, beta)

aGraph.Neighbors()
```
