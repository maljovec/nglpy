"""
    This module is meant to mirror the API from nglpy in order to create
    a drop-in replacement. Consider this class for deprecation due to
    inefficient handling of neighborhood queries.
"""
from threading import Thread
from queue import Queue, Empty

import numpy as np

from .SKLSearchIndex import SKLSearchIndex
from .Graph import Graph, get_edge_list


class ProbabilisticGraph(Graph):
    """ A probabilistic neighborhood graph that represents an uncertain
    connectivity of a given data matrix.

    Attributes:
        None
    """

    def __init__(
        self, steepness=3, index=None, max_neighbors=-1, relaxed=False, beta=1
    ):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
            index (string): A nearest neighbor index structure which can
                be queried and pruned
            max_neighbors (int): The maximum number of neighbors to
                associate with any single point in the dataset.
            relaxed (bool): Whether the relaxed ERG should be computed
            beta (float): Defines the shape of the beta skeleton
            p (float): The Lp-norm to use in computing the shape
            discrete_steps (int): The number of steps to use if using
                the discrete version. -1 (default) signifies to use the
                continuous algorithm.
            query_size (int): The number of points to process with each
                call to the GPU, this should be computed based on
                available resources
        """
        self.steepness = steepness
        self.seed = 0
        self.probabilities = None
        super(ProbabilisticGraph, self).__init__(
            index=index, max_neighbors=max_neighbors, relaxed=relaxed,
            beta=beta
        )

    def reseed(self, seed):
        self.seed = seed

    def populate(self):
        if self.probabilities is None:
            count = self.X.shape[0]
            working_set = np.array(range(count))
            distances, edges = self.nn_index.search(
                working_set, self.max_neighbors
            )

            probabilities = Graph.probability(
                self.X,
                edges,
                steepness=self.steepness,
                relaxed=self.relaxed,
                beta=self.beta,
            )

            self.probabilities = probabilities
            self.edges = edges
            self.distances = distances

        np.random.seed(self.seed)
        mask = np.random.binomial(1, 1 - self.probabilities).astype(bool)
        self.realized_edges = np.copy(self.edges)
        self.realized_edges[mask] = -1
        valid_edges = get_edge_list(self.realized_edges, self.distances)
        for edge in valid_edges:
            self.edge_list.put(edge)

    def neighbors(self, i):
        nn = []
        for value in self.realized_edges[i]:
            if value != -1:
                nn.append(value)
        return nn
