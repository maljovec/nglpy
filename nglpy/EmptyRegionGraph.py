"""
    This module abstracts some of the ugliness of interfacing with C++
    from Python by providing an easy-to-use interface to the
    Neighborhood Graph Library (NGL) originally developed by Carlos
    Correa.
"""
from .ngl import nglGraph, vectorInt, vectorDouble
import sklearn.neighbors


class EmptyRegionGraph(nglGraph):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be
    documented inline with the attribute's declaration (see __init__
    method below).

    Properties created with the ``@property`` decorator should be
    documented in the property's getter method.

    Attributes:
        None
    """

    def __init__(self,
                 index=None,
                 max_neighbors=-1,
                 relaxed=False,
                 beta=1,
                 p=2.0,
                 discrete_steps=-1,
                 template_function=None,
                 query_size=None,
                 cached=True):
        self.max_neighbors = max_neighbors
        self.relaxed = relaxed
        self.beta = beta
        self.p = p

    def build(self, X):
        cols = 0
        rows = len(X)
        if rows > 0:
            cols = len(X[0])

        flattened_X = [xij for Xi in X for xij in Xi]

        if self.max_neighbors <= 0:
            self.max_neighbors = rows - 1

        if self.max_neighbors >= rows:
            # Let the C++ side worry about this, do not build the knn in
            # python for a fully connected graph
            edges = vectorInt()
        else:
            knn = sklearn.neighbors.NearestNeighbors(self.max_neighbors)
            knn.fit(X)
            edges = knn.kneighbors(X, return_distance=False)

            # use pairs to prevent duplicates
            pairs = []
            for e1 in range(0, edges.shape[0]):
                for col in range(0, edges.shape[1]):
                    e2 = edges.item(e1, col)
                    if e1 != e2:
                        pairs.append((e1, e2))

            # As seen here: https://bit.ly/1pUtpLh
            seen = set()
            pairs = [
                x
                for x in pairs
                if not (x in seen or x[::-1] in seen or seen.add(x))
            ]
            edgeList = []
            for edge in pairs:
                edgeList.append(edge[0])
                edgeList.append(edge[1])
            edges = vectorInt(edgeList)

        if self.relaxed:
            graph = "relaxed beta skeleton"
        else:
            graph = "beta skeleton"

        super(EmptyRegionGraph, self).__init__(
            vectorDouble(flattened_X),
            rows,
            cols,
            graph,
            self.max_neighbors,
            self.beta,
            edges,
            False,
        )

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
            return dict(self.full_graph())
        else:
            return list(super(EmptyRegionGraph, self).get_neighbors(int(idx)))
