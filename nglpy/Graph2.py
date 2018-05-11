
########################################################################
# Software License Agreement (BSD License)                             #
#                                                                      #
# Copyright 2018 University of Utah                                    #
# Scientific Computing and Imaging Institute                           #
# 72 S Central Campus Drive, Room 3750                                 #
# Salt Lake City, UT 84112                                             #
#                                                                      #
# THE BSD LICENSE                                                      #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# 1. Redistributions of source code must retain the above copyright    #
#    notice, this list of conditions and the following disclaimer.     #
# 2. Redistributions in binary form must reproduce the above copyright #
#    notice, this list of conditions and the following disclaimer in   #
#    the documentation and/or other materials provided with the        #
#    distribution.                                                     #
# 3. Neither the name of the copyright holder nor the names of its     #
#    contributors may be used to endorse or promote products derived   #
#    from this software without specific prior written permission.     #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR #
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED       #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE   #
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY       #
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL   #
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE    #
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS        #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER #
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR      #
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN  #
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                        #
########################################################################
"""
    This module provides a pure python interface that replicates the
    functionality of the Neighborhood Graph Library (NGL) originally
    developed by Carlos Correa.
"""
import sklearn.neighbors
from sklearn.metrics.pairwise import paired_euclidean_distances
import numpy as np

from .UnionFind import UnionFind


class Graph2(object):
    """ A neighborhood graph that represents the connectivity of a given
    data matrix.
    """

    @staticmethod
    def max_distance(t, beta, p):

        xC = 0
        yC = 0

        if t > 1:
            return 0

        if beta <= 1:
            r = 1 / beta
            yC = (r**p - 1)**(1/p)
        else:
            r = beta
            xC = 1 - beta
        
        y = (r**p - (t-xC)**p)**1/p - yC

        return 0.5*y

    @staticmethod
    def connect_components(X, edges):
        components = UnionFind()
        for edge in edges:
            components.union(edge[0], edge[1])

        reps = components.sets.keys()
        while len(reps) > 1:
            if len(reps) > 1:
                print('Connected Components: {} (Graph size: {}'.format(len(reps),
                                                                        len(edges)))
            min_distance = -1
            p1 = -1
            p2 = -1

            # Walkthrough all pairs of points on different components
            # and connect the closest pair
            for k, a in enumerate(reps):
                for b in reps[k+1:]:
                    for i in components.sets[a]:
                        ai = X[i, :]
                        for j in components.sets[b]:
                            bj = X[j, :]
                            distance = np.linalg.norm(ai-bj)

                            if min_distance == -1 or distance < min_distance:
                                p1 = i
                                p2 = j

            components.union(p1, p2)
            if p2 < p1:
                p1, p2 = p2, p1

            edges.append((p1, p2))
            reps = components.sets.keys()

        return edges

    @staticmethod
    def create_template(beta, p=2, steps=100):
        """ Method for creating a template that can be mapped to each edge in
        a graph, since the template is symmetric, it will map from one endpoint
        to the center of the edge.

        Args:
            beta (float): The beta value for the lune-based beta-skeleton
            p (positive float): The p value specifying what kind of Lp-norm to
                use to compute the shape of the lunes.
        """
        template = np.zeros(steps+1)

        if p == float("inf"):
            if beta >= 1:
                template[1:steps-1] = beta/2

            return template

        for i in range(1, steps):
            template[i] = Graph2.max_distance(i/steps, beta, p)

        return template


    @staticmethod
    def prune_edges(X, edges, beta, p, relaxed=False):
        """ Method to prune edges from an existing graph based on a specific
        beta skeleton definition

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
            edges (list of integer pairs): List of edges to prune
            beta (float): The beta value for the lune-based
                beta-skeleton
            p (float): The p value specifying what kind of Lp-norm to
                use to compute the shape of the lunes.
            max_neighbors (int): The maximum number of neighbors to
                associate with any single point in the dataset.
            edges (list): A list of pre-defined edges to prune
            connect (boolean): A flag specifying whether the data should
                be a single connected component.
        """
        steps = 49
        template = Graph2.create_template(beta, p, steps)[::-1]
        
        pruned_edges = []

        for edge in edges:
            p = X[edge[0]]
            q = X[edge[1]]
            pq = q - p
            Xp = X - p

            projections = np.dot(Xp, pq)
            distances_to_edge = paired_euclidean_distances(Xp, projections*pq)

            # First note that our template will be reversed from what the
            # create_template returns, that is 0 index is the edge midpoint and
            # step index is either endpoint. The *2*steps term extends our scale
            # to the full edge, and the -steps shifts us to where 0 is the
            # midpoint and the endpoints are at +/- steps, the numbers are then
            # cast to integers to ready them for indexing. Taking the absolute
            # value folds our scale back on itself allowing us to deal only
            # with 0-steps range of indices. Lastly, we clip all of our data to
            # the endpoints since we will always force the template to be zero
            # at these locations meaning nothing can fail the empty region
            # criteria at either endpoint.
            lookup_indices = np.clip(np.abs((projections * 2*steps - steps).astype(int)), 0, steps)
            if not len(np.where(distances_to_edge < template[lookup_indices])[0]):
                pruned_edges.append(edge)

        return pruned_edges

    def __init__(self, X, beta, p=2, max_neighbors=-1, edges=None,
                 connect=False):
        """Initialization of the graph object. This will convert all of
        the passed in parameters into parameters the C++ implementation
        of NGL can understand and then issue an external call to that
        library.

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
            beta (float): The beta value for the lune-based
                beta-skeleton
            p (float): The p value specifying what kind of Lp-norm to
                use to compute the shape of the lunes.
            max_neighbors (int): The maximum number of neighbors to
                associate with any single point in the dataset.
            edges (list): A list of pre-defined edges to prune
            connect (boolean): A flag specifying whether the data should
                be a single connected component.
        """

        rows = len(X)
        if maxN <= 0 or maxN >= rows:
            maxN = rows-1

        if edges is None:
            knnAlgorithm = sklearn.neighbors.NearestNeighbors(maxN)
            knnAlgorithm.fit(X)
            edges = knnAlgorithm.kneighbors(X, return_distance=False)

            # use pairs to prevent duplicates
            pairs = []
            for e1 in range(0, edges.shape[0]):
                    for col in range(0, edges.shape[1]):
                        e2 = edges.item(e1, col)
                        if e1 != e2:
                            pairs.append((e1, e2))
        else:
            pairs = []
            for i in range(0, len(edges), 2):
                pairs.append((edges[i], edges[i+1]))

        # As seen here:
        #  http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
        seen = set()
        pairs = [x for x in pairs if not (x in seen or x[::-1] in seen or
                                          seen.add(x))]

        self.edges = Graph2.prune_edges(X, pairs, beta, p)

        if connect:
            self.edges = Graph2.connect_components(X, self.edges)

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
        pass
