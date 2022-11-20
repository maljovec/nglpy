""" This module will test the functionality of nglpy.Graph when using
    the beta skeleton graph type
"""
import unittest
import nglpy as ngl
import os
import numpy as np


class TestGraph(unittest.TestCase):
    """ Class for testing the Gabriel graph
    """

    def setup(self):
        """ Setup function will create a fixed point set and parameter
        settings for testing different aspects of this library.
        """

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.points = np.loadtxt(os.path.join(dir_path, "data", "points.txt"))
        gold = np.loadtxt(
            os.path.join(dir_path, "data", "gold_edges_strict.txt")
        )
        self.gold_strict = set()
        for edge in gold:
            lo, hi = min(edge), max(edge)
            self.gold_strict.add((lo, hi))

        gold = np.loadtxt(
            os.path.join(dir_path, "data", "gold_edges_relaxed.txt")
        )
        self.gold_relaxed = set()
        for edge in gold:
            lo, hi = min(edge), max(edge)
            self.gold_relaxed.add((lo, hi))

    def test_strict(self):
        """
        Test Graph's ability to build knn and prune correctly in the
        strict case for both discrete and continuous algorithms.
        """
        self.setup()
        graph = ngl.Graph(
            index=None,
            max_neighbors=-1,
            relaxed=False,
            beta=1,
            p=2.0,
            discrete_steps=-1,
        )
        graph.build(self.points)

        test = set()
        for e1, e2, d in graph:
            test.add((min(e1, e2), max(e1, e2)))

        self.assertSetEqual(self.gold_strict, test)

        graph = ngl.Graph(
            index=None,
            max_neighbors=-1,
            relaxed=False,
            beta=1,
            p=2.0,
            discrete_steps=100,
        )
        graph.build(self.points)
        test = set()
        for e1, e2, d in graph:
            test.add((min(e1, e2), max(e1, e2)))

        self.assertSetEqual(self.gold_strict, test)

    def test_relaxed(self):
        """
        Test Graph's ability to build knn and prune correctly in the
        strict case for both discrete and continuous algorithms.
        """
        self.setup()
        graph = ngl.Graph(
            index=None,
            max_neighbors=-1,
            relaxed=True,
            beta=1,
            p=2.0,
            discrete_steps=-1,
        )
        graph.build(self.points)
        test = set()
        for e1, e2, d in graph:
            test.add((min(e1, e2), max(e1, e2)))

        self.assertSetEqual(self.gold_relaxed, test)

        graph = ngl.Graph(
            index=None,
            max_neighbors=-1,
            relaxed=True,
            beta=1,
            p=2.0,
            discrete_steps=100,
        )

        graph.build(self.points)
        test = set()
        for e1, e2, d in graph:
            test.add((min(e1, e2), max(e1, e2)))

        self.assertSetEqual(self.gold_relaxed, test)


if __name__ == "__main__":
    unittest.main()
