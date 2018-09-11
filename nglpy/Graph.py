"""
    This module abstracts some of the ugliness of interfacing with C++
    from Python by providing an easy-to-use interface to the
    Neighborhood Graph Library (NGL) originally developed by Carlos
    Correa.
"""
import faiss
import numpy as np


class Graph(object):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.
    """

    def __init__(self, X, index, max_neighbors=-1):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
            index (SearchIndex): An object to use for querying
            max_neighbors (int): The maximum number of neighbors to
                associate with any single point in the dataset.
        """
        tolerance = 1e-6
        N, D = X.shape
        if max_neighbors < 0:
            max_neighbors = N-1

        query_size = min(1000000, N)
        self.edges = {}
        for offset in range(0, N, query_size):
            all_dist, neighbors = index.search(X[offset:(offset+query_size)], max_neighbors)
            midpts = np.empty((query_size*max_neighbors, D), np.float32)
            for row, row_values in enumerate(neighbors):
                i = offset+row
                for col, idx in enumerate(row_values):
                    pseudo_idx = row*max_neighbors+col
                    midpts[pseudo_idx] = (X[idx] + X[i]) / 2.

            distances, closest = index.search(midpts, 1)
            for pseudo_idx, (dist, idx) in enumerate(zip(distances, closest)):
                row = pseudo_idx // max_neighbors
                col = pseudo_idx % max_neighbors

                i = offset+row
                j = neighbors[row, col]
                radius = all_dist[row, col] / 2.

                if idx in [i, j] or radius - dist < tolerance:
                    if i not in self.edges:
                        self.edges[i] = set()
                    if j not in self.edges:
                        self.edges[j] = set()

                    self.edges[i].add(j)
                    self.edges[j].add(i)

    def neighbors(self, idx=None):
        """ Returns the list of neighbors associated to a particular
            index in the dataset, if one is provided, otherwise a full
            dictionary is provided relating each index to a set of
            connected indices.

        Args:
            idx: (optional) a single index of the point in the input
                data matrix for which we want to retrieve neighbors.

        Returns:
            A list of indices connected to either the provided input
            index, or a dictionary where the keys are the indices in the
            whole dataset and the values are sets of indices connected
            to the key index.
        """
        if idx is None:
            return self.edges
        else:
            return self.edges[idx]
