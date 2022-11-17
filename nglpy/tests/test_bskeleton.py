""" This module will test the functionality of nglpy.EmptyRegionGraph
    when using the beta skeleton graph type
"""
import unittest

import nglpy


class TestBSkeleton(unittest.TestCase):
    """Class for testing the Gabriel graph"""

    def setup(self):
        """Setup function will create a fixed point set and parameter
        settings for testing different aspects of this library.

        Test graph shape:

         3----2
             /
            1
          /  |
        0     4

        However, the pruned edges should result in this graph:
        expected_graph = {0: (1, 2), 1: (0, 3, 4), 2: (0, 3, 4),
                          3: (1, 2), 4: (1, 2)}

         3----2
          | /  |
           /1  |
         //  | |
        0     4

        The relaxed version should look like:

         3----2
         |   /
        |   1
        | /  |
        0     4

        """

        self.points = [
            [0.360502, 0.535494],
            [0.476489, 0.560185],
            [0.503125, 0.601218],
            [0.462382, 0.666667],
            [0.504702, 0.5],
        ]
        self.max_neighbors = 4
        self.beta = 1
        self.relaxed = False
        self.p = 2.0
        self.discrete_steps = -1
        self.edges = [0, 1, 0, 2, 0, 3, 0, 4, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4]

    def test_neighbors(self):
        """Tests the neighbors function in both settings, that is where
        an index is supplied and when it is not. This does not use
        an input neighborhood graph, thus NGL must prune the
        complete graph in this case.
        """
        self.setup()
        graph_rep = nglpy.EmptyRegionGraph(
            max_neighbors=self.max_neighbors,
            beta=self.beta,
            relaxed=self.relaxed,
            p=self.p,
            discrete_steps=self.discrete_steps,
        )
        graph_rep.build(self.points)
        expected_graph = {0: (1,), 1: (0, 2, 4), 2: (1, 3), 3: (2,), 4: (1,)}

        for i in range(len(self.points)):
            expected = list(expected_graph[i])
            actual = sorted(graph_rep.neighbors(i))
            msg = "\nNode {} Connectivity:".format(i)
            msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
            self.assertEqual(expected, actual, msg)

        self.assertEqual(graph_rep.neighbors(), expected_graph)

    # def test_neighbors_with_edges(self):
    #     """ Tests the neighbors function in both settings, that is where
    #         an index is supplied and when it is not. A user supplied
    #         sub-graph is used in this case. Since, this sub-graph prunes
    #         valid edges, then we should see those edges removed from the
    #         actual graph.
    #     """
    #     self.setup()
    #     # TODO: get this variable into a pre-built graph index: self.edges
    #     graph_rep = nglpy.EmptyRegionGraph(max_neighbors=self.max_neighbors,
    #                                        beta=self.beta,
    #                                        relaxed=self.relaxed,
    #                                        p=self.p,
    #                                        discrete_steps=self.discrete_steps)

    #     graph_rep.build(self.points)
    #     expected_graph = {
    #         0: (1, 2),
    #         1: (0, 3, 4),
    #         2: (0, 3, 4),
    #         3: (1, 2),
    #         4: (1, 2),
    #     }

    #     for i in range(len(self.points)):
    #         expected = list(expected_graph[i])
    #         actual = sorted(graph_rep.neighbors(i))
    #         msg = "\nNode {} Connectivity:".format(i)
    #         msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
    #         self.assertEqual(expected, actual, msg)

    #     self.assertEqual(graph_rep.neighbors(), expected_graph)

    def test_RelaxedNeighborhood(self):
        """Tests the neighbors function in both settings, that is where
        an index is supplied and when it is not. This does not use
        an input neighborhood graph, thus NGL must prune the
        complete graph in this case.
        """
        self.setup()
        graph_rep = nglpy.EmptyRegionGraph(
            max_neighbors=0,
            beta=self.beta,
            relaxed=True,
            p=self.p,
            discrete_steps=self.discrete_steps,
        )
        graph_rep.build(self.points)
        expected_graph = {
            0: (1, 3),
            1: (0, 2, 4),
            2: (1, 3),
            3: (0, 2),
            4: (1,),
        }

        for i in range(len(self.points)):
            expected = list(expected_graph[i])
            actual = sorted(graph_rep.neighbors(i))
            msg = "\nNode {} Connectivity:".format(i)
            msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
            self.assertEqual(expected, actual, msg)

        self.assertEqual(graph_rep.neighbors(), expected_graph)

    def test_empty(self):
        """Tests handling of empty data, we just want to make sure nothing
        breaks terribly.
        """
        self.setup()
        graph_rep = nglpy.EmptyRegionGraph(
            max_neighbors=self.max_neighbors,
            beta=self.beta,
            relaxed=True,
            p=self.p,
            discrete_steps=self.discrete_steps,
        )
        graph_rep.build([])
        expected_graph = {}

        self.assertEqual(graph_rep.neighbors(), expected_graph)


# TODO: Test if the kmax parameter ever gets used in this version of
# NGL, since it does not require ANN, I am assuming it does a brute
# force search of the edges if we don't provide them. We should remove
# that parameter from NGL if that is the case


if __name__ == "__main__":
    unittest.main()
