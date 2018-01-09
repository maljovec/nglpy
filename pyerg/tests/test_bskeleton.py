##############################################################################
 # Software License Agreement (BSD License)                                   #
 #                                                                            #
 # Copyright 2018 University of Utah                                          #
 # Scientific Computing and Imaging Institute                                 #
 # 72 S Central Campus Drive, Room 3750                                       #
 # Salt Lake City, UT 84112                                                   #
 #                                                                            #
 # THE BSD LICENSE                                                            #
 #                                                                            #
 # Redistribution and use in source and binary forms, with or without         #
 # modification, are permitted provided that the following conditions         #
 # are met:                                                                   #
 #                                                                            #
 # 1. Redistributions of source code must retain the above copyright          #
 #    notice, this list of conditions and the following disclaimer.           #
 # 2. Redistributions in binary form must reproduce the above copyright       #
 #    notice, this list of conditions and the following disclaimer in the     #
 #    documentation and/or other materials provided with the distribution.    #
 # 3. Neither the name of the copyright holder nor the names of its           #
 #    contributors may be used to endorse or promote products derived         #
 #    from this software without specific prior written permission.           #
 #                                                                            #
 # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       #
 # IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  #
 # OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    #
 # IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           #
 # INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   #
 # NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  #
 # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      #
 # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   #
 # THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          #
 ##############################################################################
"""
    This module will test the functionality of pyerg.Graph when using the
    beta skeleton graph type
"""
from unittest import TestCase

import pyerg

class TestBSkeleton(TestCase):
    """
    Class for testing the Gabriel graph
    """

    def setup(self):
        """
        Setup function will create a fixed point set and parameter settings for
        testing different aspects of this library.

        Test graph shape:

        3 - 2
          /   \
        0       1

        However, the pruned edges should result in this graph:

        3 - 2
        |     \
        0 ----- 1

        """

        self.points = [[0, 0], [2, 0], [1, 1], [0, 1]]
        self.max_neighbors = 3
        self.beta = 1
        self.graph = 'beta skeleton'
        self.edges = [0, 1, 0, 3, 1, 2, 1, 3, 2, 3]

    def test_Neighbors(self):
        """
        Tests the Neighbors function in both settings, that is where an index
        is supplied and when it is not. This does not use an input neighborhood
        graph, thus NGL must prune the complete graph in this case.
        """
        self.setup()
        graph_rep = pyerg.Graph(self.points, self.graph, self.max_neighbors, self.beta)
        expected_graph = {0: (3, ), 1: (2, ), 2: (1, 3), 3: (0, 2)}

        for i in range(len(self.points)):
            expected = list(expected_graph[i])
            actual = sorted(graph_rep.Neighbors(i))
            msg = '\nNode {} Connectivity:\n\texpected: {}\n\tactual: {} '.format(i, expected, actual)
            self.assertEqual(expected, actual, msg)

        self.assertEqual(graph_rep.Neighbors(), expected_graph)

    def test_Neighbors_with_edges(self):
        """
        Tests the Neighbors function in both settings, that is where an index
        is supplied and when it is not. A user supplied sub-graph is used in
        this case. Since, this sub-graph prunes valid edges, then we should see
        those edges removed from the actual graph.
        """
        self.setup()
        graph_rep = pyerg.Graph(self.points, self.graph, self.max_neighbors, self.beta, self.edges)

        expected_graph = {0: (1,3), 1: (0,2), 2: (1, 3), 3: (0, 2)}

        for i in range(len(self.points)):
            expected = list(expected_graph[i])
            actual = sorted(graph_rep.Neighbors(i))
            msg = 'Node {} Connectivity:\n\texpected: {}\n\tactual: {} '.format(i, expected, actual)
            self.assertEqual(expected, actual, msg)

        self.assertEqual(graph_rep.Neighbors(), expected_graph)

if __name__ == '__main__':
    unittest.main()
