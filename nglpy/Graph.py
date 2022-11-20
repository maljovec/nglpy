"""
    This module abstracts some of the ugliness of interfacing with C++
    from Python by providing an easy-to-use interface to the
    Neighborhood Graph Library (NGL) originally developed by Carlos
    Correa.
"""
# from threading import Thread
from queue import Queue, Empty

import numpy as np

from .SKLSearchIndex import SKLSearchIndex


def logistic_function(x, r, steepness=3):
    with np.errstate(divide="ignore"):
        k = steepness / r
        return 1 / (1 + np.exp(-k * (x - r)))


def paired_lpnorms(A, B, p=2):
    """ Method to compute the paired Lp-norms between two sets of
        points. Note, A and B should be the same shape.

    Args:
        A (MxN matrix): A collection of points
        B (MxN matrix): A collection of points
        p (positive float): The p value specifying what kind of Lp-norm
            to use to compute the shape of the lunes.
    """
    N = A.shape[0]
    dimensionality = A.shape[1]
    norms = np.zeros(N)
    for i in range(N):
        norm = 0.0
        for k in range(dimensionality):
            norm += (A[i, k] - B[i, k]) ** p
        norms[i] = norm ** (1. / p)
    return norms


def min_distance_from_edge(t, beta, p):
    """ Using a parameterized scale from [0,1], this function will
        determine the minimum valid distance to an edge given a
        specified lune shape defined by a beta parameter for defining
        the radius and a p parameter specifying the type of Lp-norm the
        lune's shape will be defined in.

    Args:
        t (float): the parameter value defining how far into the edge we
            are. 0 means we are at one of the endpoints, 1 means we are
            at the edge's midpoint.
        beta (float): The beta value for the lune-based beta-skeleton
        p (float): The p value specifying which Lp-norm to use to
            compute the shape of the lunes. A negative value will be
            used to represent the inf norm

    """
    xC = 0
    yC = 0
    if t > 1:
        return 0
    if beta <= 1:
        r = 1 / beta
        yC = (r ** p - 1) ** (1. / p)
    else:
        r = beta
        xC = 1 - beta
    y = (r ** p - (t - xC) ** p) ** (1. / p) - yC
    return 0.5 * y


def create_template(beta, p=2, steps=100):
    """ Method for creating a template that can be mapped to each edge
        in a graph, since the template is symmetric, it will map from
        one endpoint to the center of the edge.

    Args:
        beta (float [0,1]): The beta value for the lune-based beta
            skeleton
        p (positive float): The p value specifying which Lp-norm to use
            to compute the shape of the lunes.
    """
    template = np.zeros(steps + 1)
    if p < 0:
        if beta >= 1:
            template[:-1] = beta / 2
        return template
    for i in range(steps):
        template[i] = min_distance_from_edge(i / steps, beta, p)
    return template


def get_edge_list(edges, distances):
    edge_list = []
    for i, row in enumerate(edges):
        for j, value in enumerate(row):
            if value != -1:
                edge_list.append((int(i), int(value), distances[i, j]))
    return edge_list


class Graph(object):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    @staticmethod
    def prune_discrete(X, edges, beta=1, lp=2, relaxed=False, steps=99):
        problem_size = edges.shape[0]
        template = create_template(beta, lp, steps)
        pruned_edges = np.zeros(shape=edges.shape) - 1
        for i in range(problem_size):
            p = X[i]
            for k in range(edges.shape[1]):
                j = edges[i, k]
                q = X[j]
                if i == j:
                    continue
                pq = q - p
                edge_length = np.linalg.norm(pq)
                if relaxed:
                    subset = np.setdiff1d(pruned_edges[i, :k], [-1]).astype(
                        np.int64
                    )
                else:
                    subset = np.concatenate((edges[i], edges[j])).astype(
                        np.int64
                    )

                Xp = X[subset] - p
                projections = np.dot(Xp, pq) / (edge_length ** 2)
                lookup_indices = np.abs(
                    np.rint(steps * (projections * 2 - 1)).astype(np.int64)
                )
                temp_indices = np.logical_and(
                    lookup_indices >= 0, lookup_indices <= steps
                )
                valid_indices = np.nonzero(temp_indices)[0]
                temp = np.atleast_2d(projections[valid_indices]).T * pq
                distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)
                points_in_region = np.nonzero(
                    distances_to_edge
                    < edge_length * template[lookup_indices[valid_indices]]
                )[0]
                if len(points_in_region) == 0:
                    pruned_edges[i, k] = j
        return pruned_edges

    @staticmethod
    def prune(X, edges, beta=1, lp=2, relaxed=False):
        problem_size = edges.shape[0]
        pruned_edges = np.zeros(shape=edges.shape) - 1
        for i in range(problem_size):
            p = X[i]
            for k in range(edges.shape[1]):
                j = edges[i, k]
                q = X[j]
                if i == j:
                    continue
                pq = q - p
                edge_length = np.linalg.norm(pq)
                if relaxed:
                    subset = np.setdiff1d(pruned_edges[i, :k], [-1]).astype(
                        np.int64
                    )
                else:
                    subset = np.concatenate((edges[i], edges[j])).astype(
                        np.int64
                    )
                Xp = X[subset] - p
                projections = np.dot(Xp, pq) / (edge_length ** 2)
                temp_indices = np.logical_and(
                    projections > 0.,
                    np.logical_and(
                        projections < 1.,
                        np.logical_and(subset != i, subset != j),
                    ),
                )
                valid_indices = np.nonzero(temp_indices)[0]
                temp = np.atleast_2d(projections[valid_indices]).T * pq
                min_distances = np.zeros(len(valid_indices))
                for idx, t in enumerate(projections[valid_indices]):
                    min_distances[idx] = (
                        min_distance_from_edge(abs(2 * t - 1), beta, lp)
                        * edge_length
                    )
                distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)
                points_in_region = np.nonzero(
                    distances_to_edge < min_distances
                )[0]
                if len(points_in_region) == 0:
                    pruned_edges[i, k] = j
        return pruned_edges

    @staticmethod
    def probability_discrete(
        X, edges, beta=1, lp=2, relaxed=False, steepness=3, steps=99
    ):
        problem_size = edges.shape[0]
        template = create_template(beta, lp, steps)
        probabilities = np.zeros(shape=edges.shape)
        for i in range(problem_size):
            p = X[i]
            for k in range(edges.shape[1]):
                j = edges[i, k]
                q = X[j]
                if i == j:
                    continue
                pq = q - p
                edge_length = np.linalg.norm(pq)
                if relaxed:
                    subset = edges[i, :k]
                    subset = subset[np.where(probabilities[i, :k] > 1e-6)]
                else:
                    subset = np.concatenate((edges[i], edges[j]))

                Xp = X[subset] - p
                projections = np.dot(Xp, pq) / (edge_length ** 2)
                lookup_indices = np.abs(
                    np.rint(steps * (projections * 2 - 1)).astype(np.int64)
                )
                temp_indices = np.logical_and(
                    lookup_indices >= 0, lookup_indices <= steps
                )
                valid_indices = np.nonzero(temp_indices)[0]
                temp = np.atleast_2d(projections[valid_indices]).T * pq
                distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)
                probabilities[i, k] = np.min(
                    logistic_function(
                        distances_to_edge,
                        edge_length * template[lookup_indices[valid_indices]],
                        steepness,
                    )
                )
        return probabilities

    @staticmethod
    def probability(X, edges, beta=1, lp=2, relaxed=False, steepness=3):
        problem_size = edges.shape[0]
        probabilities = np.zeros(shape=edges.shape)
        for i in range(problem_size):
            p = X[i]
            for k in range(edges.shape[1]):
                j = edges[i, k]
                q = X[j]
                if i == j:
                    continue
                pq = q - p
                edge_length = np.linalg.norm(pq)
                if relaxed:
                    subset = edges[i, :k]
                    subset = subset[np.where(probabilities[i, :k] > 1e-6)]
                else:
                    subset = np.concatenate((edges[i], edges[j]))
                Xp = X[subset] - p
                projections = np.dot(Xp, pq) / (edge_length ** 2)
                temp_indices = np.logical_and(
                    projections > 0.,
                    np.logical_and(
                        projections < 1.,
                        np.logical_and(subset != i, subset != j),
                    ),
                )
                valid_indices = np.nonzero(temp_indices)[0]
                temp = np.atleast_2d(projections[valid_indices]).T * pq
                min_distances = np.zeros(len(valid_indices))
                for idx, t in enumerate(projections[valid_indices]):
                    min_distances[idx] = (
                        min_distance_from_edge(abs(2 * t - 1), beta, lp)
                        * edge_length
                    )
                distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)

                probabilities[i, k] = np.min(
                    logistic_function(
                        distances_to_edge, min_distances, steepness
                    )
                )
        return probabilities

    def __init__(self, index=None, max_neighbors=-1, relaxed=False, beta=1, p=2, discrete_steps=-1):
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
        """
        self.X = np.array([[]])
        self.max_neighbors = max_neighbors
        self.relaxed = relaxed
        self.beta = beta
        self.edges = None
        if index is None:
            self.nn_index = SKLSearchIndex()
        else:
            self.nn_index = index
        self.needs_reset = False
        self.lp = p
        self.discrete_steps = discrete_steps

    def build(self, X):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
        """
        self.X = np.array(X)
        N = len(self.X)

        if self.max_neighbors < 0:
            self.max_neighbors = min(1000, N)
        else:
            self.max_neighbors = self.max_neighbors

        self.nn_index.fit(self.X)

        self.edge_list = Queue(self.X.shape[0] * self.max_neighbors)
        self.needs_reset = False

        # self.worker_thread = Thread(target=self.populate, daemon=True)
        # self.worker_thread.start()
        self.populate()

    def populate(self):
        if self.edges is None:
            count = self.X.shape[0]
            working_set = np.array(range(count))
            distances, edges = self.nn_index.search(
                working_set, self.max_neighbors
            )
            edges = Graph.prune(
                self.X, edges, relaxed=self.relaxed, beta=self.beta, lp=self.lp
            )
            self.edges = edges
            self.distances = distances
        valid_edges = get_edge_list(self.edges, self.distances)
        for edge in valid_edges:
            self.edge_list.put(edge)

    def restart_iteration(self):
        # if not self.worker_thread.is_alive() and self.needs_reset:
        if self.needs_reset:
            self.edge_list.queue.clear()
            self.needs_reset = False
            # self.worker_thread = Thread(target=self.populate, daemon=True)
            # self.worker_thread.start()
            self.populate()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.needs_reset:
            self.restart_iteration()
        # while not self.edge_list.empty() or self.worker_thread.is_alive():
        while not self.edge_list.empty():
            try:
                next_edge = self.edge_list.get(timeout=1)
                return next_edge
            except Empty:
                pass
        self.needs_reset = True
        raise StopIteration

    def full_graph(self):
        neighborhoods = {}
        for (p, q, d) in self:
            p = int(p)
            q = int(q)
            if p not in neighborhoods:
                neighborhoods[p] = set()
            if q not in neighborhoods:
                neighborhoods[q] = set()
            neighborhoods[p].add(q)
            neighborhoods[q].add(p)
        return neighborhoods

    def neighbors(self, i):
        nn = []
        for value in self.edges[i]:
            if value != -1:
                nn.append(value)
        return nn
