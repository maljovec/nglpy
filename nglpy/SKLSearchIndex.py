"""
    A generic Search index structure that specifies the bare minimum
    functionality needed by any one implementation of an approximate
    k nearest neighbor structure
"""
import numpy as np
from sklearn import neighbors
from .SearchIndex import SearchIndex
from .utils import *


class SKLSearchIndex(SearchIndex):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        """ Initializes the underlying algorithm with any user-provided
            parameters
        """
        self.index = neighbors.NearestNeighbors(**kwargs)

    def fit(self, X):
        """ Will build any supporting data structures if necessary given
            the data stored in X
        """
        self.X = X
        self.index.fit(self.X)

    def search(self, idx, k, return_distance=True):
        """ Returns the list of neighbors associated to one or more
            poiints in the dataset.

        Args:
            idx: one or more indices in X for which we want to retrieve
                 neighbors.

        Returns:
            A numpy array of the k nearest neighbors to each input point

            A numpy array specifying the distances to each neighbor
        """

        output = self.index.kneighbors(self.X[idx, :],
                                       k,
                                       return_distance=return_distance)
        if return_distance:
            distance_matrix = np.array(output[0], dtype=f32)
            edge_matrix = np.array(output[1], dtype=i32)
            return distance_matrix, edge_matrix
        else:
            edge_matrix = np.array(output, dtype=i32)
            return edge_matrix
