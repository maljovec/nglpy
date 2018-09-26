"""
    A generic Search index structure that specifies the bare minimum
    functionality needed by any one implementation of an approximate
    k nearest neighbor structure
"""
from abc import ABC, abstractmethod
from .utils import *


class SearchIndex(ABC):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self, **kwargs):
        """ Initializes the underlying algorithm with any user-provided
            parameters
        """
        pass

    @abstractmethod
    def fit(self, X):
        """ Will build any supporting data structures if necessary given
            the data stored in X
        """
        pass

    @abstractmethod
    def search(self, idx, k, return_distance=True):
        """ Returns the list of neighbors associated to one or more
            poiints in the dataset.

        Args:
            idx: one or more indices in X for which we want to retrieve
                 neighbors.
            k: the maximum number of neighbors to return
            return_distance: boolean flag specifying whether distances
                             should be included in the return tuple

        Returns:
            A numpy array of the k nearest neighbors to each input point

            A numpy array specifying the distances to each neighbor
        """
        pass
