"""
    This module abstracts some of the ugliness of interfacing with C++
    from Python by providing an easy-to-use interface to the
    Neighborhood Graph Library (NGL) originally developed by Carlos
    Correa.
"""
from threading import Thread
from queue import Queue, Empty

import nglpy as ngl
import numpy as np

from .utils import f32, i32
from .SKLSearchIndex import SKLSearchIndex


class Graph(object):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.

    Attributes:
        None
    """

    def __init__(self,
                 X,
                 index=None,
                 max_neighbors=-1,
                 relaxed=False,
                 beta=1):
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
        self.X = np.array(X, dtype=f32)
        N = len(self.X)

        if max_neighbors < 0:
            self.max_neighbors = min(1000, N)
        else:
            self.max_neighbors = max_neighbors

        self.relaxed = relaxed
        self.beta = beta

        if index is None:
            self.nn_index = SKLSearchIndex()
        else:
            self.nn_index = index
        self.nn_index.fit(self.X)

        self.edge_list = Queue(self.X.shape[0]*self.max_neighbors)
        self.needs_reset = False

        self.worker_thread = Thread(target=self.populate, daemon=True)
        self.worker_thread.start()

    def populate(self):
        count = self.X.shape[0]
        working_set = np.array(range(count))
        edges = self.nn_index.search(working_set, self.max_neighbors, False)
        edges = ngl.prune(self.X, edges, relaxed=self.relaxed, beta=self.beta)
        for edge in edges:
            self.edge_list.put(edge)

    def restart_iteration(self):
        if not self.worker_thread.is_alive() and self.needs_reset:
            self.edge_list.queue.clear()
            self.needs_reset = False
            self.worker_thread = Thread(target=self.populate, daemon=True)
            self.worker_thread.start()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.needs_reset:
            self.restart_iteration()
        while not self.edge_list.empty() or self.worker_thread.is_alive():
            try:
                next_edge = self.edge_list.get(timeout=1)
                return next_edge
            except Empty:
                pass
        self.needs_reset = True
        raise StopIteration
