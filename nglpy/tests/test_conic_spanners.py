""" This module will test the functionality of nglpy.Graph when using the
    conic spanner graphs (Yao and Θ) type
"""
import unittest
import nglpy
import math
import samplers
import numpy as np


class TestConics(unittest.TestCase):
    """ Class for testing the Yao and Θ Graphs
    """

    def setup(self):
        """ Setup function will create a fixed point set and parameter
        settings for testing different aspects of the conic spanners.
        """

        # We will only test the edges of the origin, since we know what
        # they should be in both cases
        self.points = [[0., 0.]]

        # Generate the same set of vectors that the Yao and Θ graphs
        # will use in order to test the Θ graph's ability to recover
        # points exactly on the conic axes
        count = 6
        dim = 2
        vectors = samplers.SCVTSampler.generate_samples(count, dim, seed=0)

        # Now take those same vectors, rotate them by one quarter of the
        # bisecting angle of each conic section to ensure they still lie
        # well within the desired conic sections and reduce their
        # magnitudes by half. These set of points should be reported
        # as the edges of the Yao graph.
        theta = 2 * math.pi / (count * 4)
        rotation_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )
        rotated_vectors = (rotation_matrix @ vectors.T).T / 2.

        self.points = np.vstack((self.points, vectors, rotated_vectors))

        self.max_neighbors = len(self.points)
        self.k = count
        self.d = dim

    def test_theta(self):
        """ Test the Θ-Graph
        """
        self.setup()
        edges = nglpy.conic_spanners.theta_graph(
            self.points, self.d, self.k, self.max_neighbors
        )

        expected = set()
        for k in range(1, self.k+1):
            expected.add((0, k))

        actual = set()
        for e in edges:
            if 0 in e[:2]:
                if e not in expected:
                    actual.add(e[:2])
        msg = "\nNode {} Connectivity:".format(0)
        msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
        self.assertEqual(expected, actual, msg)

    def test_yao(self):
        """ Test the Yao Graph
        """
        self.setup()
        edges = nglpy.conic_spanners.yao_graph(
            self.points, self.d, self.k, self.max_neighbors
        )

        expected = set()
        for k in range(1, self.k+1):
            expected.add((0, len(self.points) - k))

        actual = set()
        for e in edges:
            if 0 in e[:2]:
                if e not in expected:
                    actual.add(e[:2])
        msg = "\nNode {} Connectivity:".format(0)
        msg += "\n\texpected: {}\n\tactual: {} ".format(expected, actual)
        self.assertEqual(expected, actual, msg)


if __name__ == "__main__":
    unittest.main()
