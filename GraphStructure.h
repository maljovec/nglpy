/******************************************************************************
 * Software License Agreement (BSD License)                                   *
 *                                                                            *
 * Copyright 2014 University of Utah                                          *
 * Scientific Computing and Imaging Institute                                 *
 * 72 S Central Campus Drive, Room 3750                                       *
 * Salt Lake City, UT 84112                                                   *
 *                                                                            *
 * THE BSD LICENSE                                                            *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * 1. Redistributions of source code must retain the above copyright          *
 *    notice, this list of conditions and the following disclaimer.           *
 * 2. Redistributions in binary form must reproduce the above copyright       *
 *    notice, this list of conditions and the following disclaimer in the     *
 *    documentation and/or other materials provided with the distribution.    *
 * 3. Neither the name of the copyright holder nor the names of its           *
 *    contributors may be used to endorse or promote products derived         *
 *    from this software without specific prior written permission.           *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    *
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   *
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  *
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   *
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          *
 ******************************************************************************/

#ifndef GRAPHSTRUCTURE_H
#define GRAPHSTRUCTURE_H

#include "ngl.h"

#include <map>
#include <vector>
#include <set>
#include <string>

/**
 * Graph Structure.
 * Induces 1-skeleton on a arbitrary dimensional point cloud using one of the
 * empty region graphs from the ngl library.
 */
template <typename T>
class GraphStructure
{
public:
  /* Here are a list of typedefs to make things more compact and readable */
  typedef std::pair<int, int> int_pair;

  typedef void (*graphFunction)(ngl::NGLPointSet<T> &points,
                                ngl::IndexType **indices, int &numEdges,
                                ngl::NGLParams<T> params);

  /**
   * Constructor that will decompose a passed in dataset, note the user passes
   * in a list of candidate edges from which it will prune accordingly using ngl
   * @param Xin flattened vector of input data in row-major order
   * @param rows int specifying the number of points in this data set
   * @param cols int specifying the number of dimensions in this data set
   * @param graph a string specifying the type of neighborhood graph to build.
   * @param maxN integer specifying the maximum number of neighbors to use in
   *        computing/pruning a neighborhood graph
   * @param beta floating point value in the range (0,2] determining the beta
   *        value used if the neighborhood type is a beta-skeleton, otherwise
   *        ignored
   * @param edgeIndices an optional list of edges specified as a flattened
   *        n-by-2 array to use as the underlying graph structure (will be
   *        pruned by ngl)
   */
  GraphStructure(std::vector<T> &Xin, int rows, int cols, std::string graph,
                 int maxN, T beta, std::vector<int> &edgeIndices,
                 bool connect = false);

  /**
   * Returns the number of input dimensions in the associated dataset
   */
  int dimension();

  /**
   * Returns the number of sample points in the associated dataset
   */
  int size();

  /**
   * Returns the maximum value attained by a specified dimension of the input
   * space of the associated dataset
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T max(int dim);

  /**
   * Returns the minimum value attained by a specified dimension of the input
   * space of the associated dataset
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T min(int dim);

  /**
   * Returns MaxX(dim)-MinX(dim)
   * @param dim integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T range(int dim);

  /**
   * Extracts the input values for a specified sample of the associated data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   * @param xi a pointer that will be updated to point at the specified data
   *        sample
   */
  void get_x(int i, T *xi);

  /**
   * Extracts the input value for a specified sample and dimension of the
   * associated data
   * @param i integer specifying the row of data where the specified sample
   *        is stored
   * @param j integer specifying the column of data where the specified input
   *        dimension is stored
   */
  T get_x(int i, int j);

  /**
   * Returns a map where the key is the index of a point and the value is a set
   * of indices that are connected to that index
   */
  std::map<int, std::set<int>> full_graph();

  /**
   * Returns a list of indices marked as neighbors to the specified sample given
   * given by "index"
   * @param index integer specifying the unique sample queried
   */
  std::set<int> get_neighbors(int index);

private:
  std::vector<std::vector<T>> X; /** Input data matrix */

  std::map<int, std::set<int>> neighbors; /** Maps a list of points
                                                     *  that are neighbors of
                                                     *  the index             */
  //////////////////////////////////////////////////////////////////////////////

  // Private Methods

  /**
   * Computes and internally stores the neighborhood graph used for
   * approximating the gradient flow behavior
   * @param data a matrix of input points upon which we will construct a
   *        neighborhood graph.
   * @param edgeIndices a vector of nonegative integer indices representing
   *        a flattened array of pre-computed edge indices to use for pruning.
   * @param type a string specifying the type of neighborhood graph to build.
   * @param beta floating point value used for the beta skeleton computation.
   * @param kmax an integer representing the maximum number of k-nearest
   *        neighbors to consider.
   * @param connect a boolean specifying whether we should enforce the graph
   *        to be a single connected component (will do a brute force search of
   *        the point samples and connect the closest points between separate
   *        components until everything is one single component)
   */
  void compute_neighborhood(std::vector<int> &edgeIndices,
                            std::string type, T beta, int &kmax,
                            bool connect = false);

  /**
   * Helper function to be called after a neighborhood has been constructed in
   * order to connect the entire domain into a single connected component.
   * This was only necessary in Sam's visualization, in theory it is fine if the
   * data is disconnected.
   */
  void connect_components(std::set<int_pair> &ngraph, int &maxCount);
};

#endif //GRAPHSTRUCTURE_H
