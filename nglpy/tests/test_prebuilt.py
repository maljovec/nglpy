""" This module will test the functionality of nglpy.PrebuiltGraph
"""
import unittest

import nglpy


class TestPrebuiltGraph(unittest.TestCase):
    """Class for testing the pre-built graph"""

    def setup(self):
        """Setup function will create a fixed point set and parameter
        settings for testing different aspects of this library.

        Test graph shape:

         3----2
             /
            1
          /  |
        0     4
        """

        self.points = [
            [0.360502, 0.535494],
            [0.476489, 0.560185],
            [0.503125, 0.601218],
            [0.462382, 0.666667],
            [0.504702, 0.5],
        ]
        self.edges = [(0, 1), (1, 2), (1, 4), (2, 3)]

    def test_neighbors(self):
        """Tests the neighbors function in both settings, that is where
        an index is supplied and when it is not. This does not use
        an input neighborhood graph, thus NGL must prune the
        complete graph in this case.
        """
        self.setup()
        graph_rep = nglpy.PrebuiltGraph(edges=self.edges)
        graph_rep.build(self.points)
        expected_graph = {0: (1,), 1: (0, 2, 4), 2: (1, 3), 3: (2,), 4: (1,)}

        for i in range(len(self.points)):
            expected = list(expected_graph[i])
            actual = sorted(graph_rep.neighbors(i))
            msg = "\nNode {} Connectivity:".format(i)
            msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
            self.assertEqual(expected, actual, msg)

        self.assertEqual(graph_rep.neighbors(), expected_graph)


# TODO: Test if the kmax parameter ever gets used in this version of
# NGL, since it does not require ANN, I am assuming it does a brute
# force search of the edges if we don't provide them. We should remove
# that parameter from NGL if that is the case


if __name__ == "__main__":
    unittest.main()
