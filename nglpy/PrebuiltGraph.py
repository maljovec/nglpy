"""
    This module abstracts some of the ugliness of interfacing with C++
    from Python by providing an easy-to-use interface to the
    Neighborhood Graph Library (NGL) originally developed by Carlos
    Correa.
"""
from nglpy import utils

from .ngl import nglGraph, vectorDouble, vectorInt


class PrebuiltGraph(nglGraph):
    """A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self, edges=None, **kwargs):
        """
        Constructor for the Prebuilt Graph class that takes a list of edges
        and as such should only be used for one dataset.

        Parameters
        ----------
        edges : list[tuple(int,int)]
            The predefined list of edges connecting a point set

        kwargs
            Forward-compatible catch-all dictionary that will allow us to throw
            a warning for every named parameter not available in this version.

        Returns
        -------
        None

        """

        utils.consume_extra_args(kwargs)
        self.edges = edges

    def build(self, X):
        rows = len(X)
        cols = len(X[0]) if rows > 0 else 0

        flattened_X = [xij for Xi in X for xij in Xi]

        # use pairs to prevent duplicates
        # As seen here: https://bit.ly/1pUtpLh
        seen = set()
        pairs = [
            x for x in self.edges if not (x in seen or x[::-1] in seen or seen.add(x))
        ]
        edgeList = []
        for edge in pairs:
            edgeList.append(int(edge[0]))
            edgeList.append(int(edge[1]))
        edges = vectorInt(edgeList)

        super(PrebuiltGraph, self).__init__(
            vectorDouble(flattened_X),
            rows,
            cols,
            "none",
            rows,
            0,
            edges,
            False,
        )

    def neighbors(self, idx=None):
        """Returns the list of neighbors associated to a particular
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
            return dict(self.full_graph())
        else:
            return list(super(PrebuiltGraph, self).get_neighbors(int(idx)))
