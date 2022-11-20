"""
    This module abstracts some of the ugliness of interfacing with C++
    from Python by providing an easy-to-use interface to the
    Neighborhood Graph Library (NGL) originally developed by Carlos
    Correa.
"""
import sklearn.neighbors

from nglpy import utils

from .ngl import nglGraph, vectorDouble, vectorInt


class EmptyRegionGraph(nglGraph):
    """A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self, *, max_neighbors=-1, relaxed=False, beta=1, p=2.0, **kwargs):
        """
        Constructor for the Empty Region Graph class that takes several keyword
        only arguments and configures a Graph object that can be applied to
        different datasets.

        This helper function will take all of the extra arguments not currently
        used by a calling function/environment, and print a warning to standard
        error letting them know they are using an unsupported feature passed into a
        function.

        Parameters
        ----------
        max_neighbors : int
            The maximum number of neighbors any one point can have. This is an
            efficiency heuristic that will limit the underlying k-nearest
            neighbor searching algorithm to only return max_neighbors edges per
            point which will then be pruned using the empty region criteria.
            This can cause a large speed-up over a full brute force edge search
            while having often little effect on the actual graph returned.
            Oftentimes, limiting the size of the empty region graph in this way
            is a benefit not a detriment for desired graph qualities.
        relaxed : bool
            Determines whether the relaxed graph (as determined by Correa and
            Lindstrom's algorithm) should be computed. A relaxed edge
            satisifies the empty region criteria from the perspective of one
            endpoint, whereas a strict edge satisfies this criteria from the
            perspective of both endpoints.
        beta : float
            A positive value working in conjunction with the p value to specify
            the size and shape of the empty region required around a valid
            edge. Roughly, beta is a size parameter and p is a shape parameter.
            For p values less than one, beta is inversely proportional to
            the size of the empty region, and for p values greater than one,
            beta is directly proportional to the size of the empty region.
            When p is one, the beta parameter is irrelevant.

            For an interactive example of how these values interact, see:
            http://www.cs.utah.edu/~maljovec/bpSkeleton.html
        p : float
            A positive value working in conjunction with the beta value to
            specify the shape and size of the empty region required around a
            valid edge. Thsi value specifies the Lp-norm to use for generating
            the empty regions. Roughly, beta is a size parameter and p is a
            shape parameter. For p values less than one, beta is inversely
            proportional to the size of the empty region, and for p values
            greater than one, beta is directly proportional to the size of the
            empty region. When p is one, the beta parameter is irrelevant.

            For an interactive example of how these values interact, see:
            http://www.cs.utah.edu/~maljovec/bpSkeleton.html

        kwargs
            Forward-compatible catch-all dictionary that will allow us to throw
            a warning for every named parameter not available in this version.

        Returns
        -------
        None

        """

        utils.consume_extra_args(kwargs)

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
            knn = sklearn.neighbors.NearestNeighbors(n_neighbors=self.max_neighbors)
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
                x for x in pairs if not (x in seen or x[::-1] in seen or seen.add(x))
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
            return list(super(EmptyRegionGraph, self).get_neighbors(int(idx)))
