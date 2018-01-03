 ##############################################################################
 # Software License Agreement (BSD License)                                   #
 #                                                                            #
 # Copyright 2014 University of Utah                                          #
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
    This module abstracts some of the ugliness of interfacing with C++ from
    Python by providing an easy-to-use interface to the Neighborhood Graph
    Library (NGL) originally developed by Carlos Correa.
"""
from .ngl import nglGraph, vectorInt, vectorDouble

class Graph(nglGraph):
    """ A neighborhood graph that represents the connectivity of a given data
    matrix.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        None
    """
    def __init__(self, X, graph, maxN, beta, edges):
        """Initialization of the graph object. This will convert all of the
        passed in parameters into parameters the C++ implementation of NGL
        can understand and then issue an external call to that library.

        Args:
            X (matrix): The data matrix for which we will be determining
                connectivity.
            graph (string): The type of graph to construct.
            maxN (int): The maximum number of neighbors to associate with any
                single point in the dataset.
            beta (float): Only relevant when the graph type is a "beta skeleton"
            edges (list): A list of pre-defined edges to prune
        """
        if edges is None:
            edges = vectorInt()
        else:
            edges = vectorInt(edges)

        super(Graph, self).__init__(vectorDouble(X.flatten()), X.shape[0], X.shape[1], graph, maxN, beta, edges)

    def Neighbors(self, idx):
        """Returns the list of neighbors associated to a particular indes in the
        dataset.

        Args:
            idx: The index of the point in the input data matrix for which we
                want to retrieve neighbors.

        Returns:
            A list of indices connected to the input index.

        """
        return list(super(Graph, self).Neighbors(int(idx)))