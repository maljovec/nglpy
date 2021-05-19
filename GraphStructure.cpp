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

#include "GraphStructure.h"
#include "UnionFind.h"

#include <algorithm>
#include <utility>
#include <limits>
#include <cstdlib>
#include <string>
#include <sstream>

template <typename T>
void GraphStructure<T>::compute_neighborhood(std::vector<int> &edgeIndices,
                                             std::string type,
                                             T beta,
                                             int &kmax,
                                             bool connect)
{
  int numPts = size();
  int dims = dimension();

  T *pts = new T[numPts * dims];
  for (int i = 0; i < numPts; i++)
    for (int d = 0; d < dims; d++)
      pts[i * dims + d] = X[d][i];

  ngl::Geometry<T>::init(dims);
  if (kmax < 0)
    kmax = numPts - 1;

  ngl::NGLPointSet<T> *P;
  ngl::NGLParams<T> params;
  params.param1 = beta;
  params.iparam0 = kmax;
  ngl::IndexType *indices = NULL;
  int numEdges = 0;

  if (edgeIndices.size() > 0)
  {
    P = new ngl::prebuiltNGLPointSet<T>(pts, numPts, edgeIndices);
  }
  else
  {
    P = new ngl::NGLPointSet<T>(pts, numPts);
  }
  delete[] pts;

  std::map<std::string, graphFunction> graphAlgorithms;
  graphAlgorithms["approximate knn"] = ngl::getKNNGraph<T>;
  graphAlgorithms["beta skeleton"] = ngl::getBSkeleton<T>;
  graphAlgorithms["relaxed beta skeleton"] = ngl::getRelaxedBSkeleton<T>;
  graphAlgorithms["diamond graph"] = ngl::getDiamondGraph<T>;
  graphAlgorithms["relaxed diamond graph"] = ngl::getRelaxedDiamondGraph<T>;
  //As it turns out, NGL's KNN graph assumes the input data is a KNN and so, is
  // actually just a pass through method that passes every input edge. We can
  // leverage this to accept "none" graphs.
  graphAlgorithms["none"] = ngl::getKNNGraph<T>;

  if (graphAlgorithms.find(type) == graphAlgorithms.end())
  {
    //TODO
    //These checks can probably be done upfront, so as not to waste computation
    std::cerr << "Invalid graph type: " << type << std::endl;
    exit(1);
  }

  graphAlgorithms[type](*P, &indices, numEdges, params);
  std::stringstream ss;
  ss << "\t\t(Edges: " << numEdges << ")" << std::endl;

  delete P;

  std::vector<int> nextNeighborId(numPts, 0);

  neighbors.clear();
  for (int i = 0; i < numPts; i++)
    neighbors[i] = std::set<int>();

  if (connect)
  {
    std::set<std::pair<int, int>> ngraph;
    for (int i = 0; i < numEdges; i++)
    {
      std::pair<int, int> edge;
      if (indices[2 * i + 0] > indices[2 * i + 1])
      {
        edge.first = indices[2 * i + 1];
        edge.second = indices[2 * i + 0];
      }
      else
      {
        edge.first = indices[2 * i + 0];
        edge.second = indices[2 * i + 1];
      }
      ngraph.insert(edge);
    }

    connect_components(ngraph, kmax);

    for (std::set<std::pair<int, int>>::iterator it = ngraph.begin();
         it != ngraph.end(); it++)
    {
      int i1 = it->first;
      int i2 = it->second;

      neighbors[i1].insert(i2);
      neighbors[i2].insert(i1);
    }
  }
  else
  {
    for (int i = 0; i < numEdges; i++)
    {
      int i1 = indices[2 * i + 0];
      int i2 = indices[2 * i + 1];

      neighbors[i1].insert(i2);
      neighbors[i2].insert(i1);
    }
  }
}

template <typename T>
GraphStructure<T>::GraphStructure(std::vector<T> &Xin, int rows, int cols,
                                  std::string graph, int maxN, T beta,
                                  std::vector<int> &edgeIndices, bool connect)
{

  int M = cols;
  int N = rows;

  X = std::vector<std::vector<T>>(M, std::vector<T>(N, 0));

  for (int n = 0; n < N; n++)
  {
    for (int m = 0; m < M; m++)
    {
      X[m][n] = Xin[n * M + m];
    }
  }

  int kmax = maxN;

  compute_neighborhood(edgeIndices, graph, beta, kmax, connect);
}

template <typename T>
void GraphStructure<T>::connect_components(std::set<int_pair> &ngraph,
                                           int &maxCount)
{
  UnionFind connectedComponents;
  for (int i = 0; i < size(); i++)
    connectedComponents.MakeSet(i);

  for (std::set<int_pair>::iterator iter = ngraph.begin();
       iter != ngraph.end();
       iter++)
  {
    connectedComponents.Union(iter->first, iter->second);
  }

  int numComponents = connectedComponents.CountComponents();
  std::vector<int> reps;
  connectedComponents.GetComponentRepresentatives(reps);
  if (numComponents > 1)
  {
    std::stringstream ss;
    ss << "Connected Components: " << numComponents << "(Graph size: "
       << ngraph.size() << ")" << std::endl;
    for (unsigned int i = 0; i < reps.size(); i++)
      ss << reps[i] << " ";
  }

  while (numComponents > 1)
  {
    //Get each representative of a component and store each
    // component into its own set
    std::vector<int> reps;
    connectedComponents.GetComponentRepresentatives(reps);
    std::vector<int> *components = new std::vector<int>[reps.size()];
    for (unsigned int i = 0; i < reps.size(); i++)
      connectedComponents.GetComponentItems(reps[i], components[i]);

    //Determine closest points between all pairs of components
    double minDistance = -1;
    int p1 = -1;
    int p2 = -1;

    for (unsigned int a = 0; a < reps.size(); a++)
    {
      for (unsigned int b = a + 1; b < reps.size(); b++)
      {
        for (unsigned int i = 0; i < components[a].size(); i++)
        {
          int AvIdx = components[a][i];
          std::vector<T> ai;
          for (int d = 0; d < dimension(); d++)
            ai.push_back(X[d][AvIdx]);
          for (unsigned int j = 0; j < components[b].size(); j++)
          {
            int BvIdx = components[b][j];
            std::vector<T> bj;
            for (int d = 0; d < dimension(); d++)
              bj.push_back(X[d][BvIdx]);

            T distance = 0;
            for (int d = 0; d < dimension(); d++)
              distance += (ai[d] - bj[d]) * (ai[d] - bj[d]);
            if (minDistance == -1 || distance < minDistance)
            {
              minDistance = distance;
              p1 = components[a][i];
              p2 = components[b][j];
            }
          }
        }
      }
    }

    //Merge
    connectedComponents.Union(p1, p2);
    if (p1 < p2)
    {
      int_pair edge = std::make_pair(p1, p2);
      ngraph.insert(edge);
    }
    else
    {
      int_pair edge = std::make_pair(p1, p2);
      ngraph.insert(edge);
    }

    //Recompute
    numComponents = connectedComponents.CountComponents();
    if (numComponents > 1)
    {
      std::stringstream ss;
      ss << "Connected Components: " << numComponents << "(Graph size: "
         << ngraph.size() << ")" << std::endl;
    }

    delete[] components;
  }
  int *counts = new int[size()];
  for (int i = 0; i < size(); i++)
    counts[i] = 0;

  for (std::set<int_pair>::iterator it = ngraph.begin();
       it != ngraph.end();
       it++)
  {
    counts[it->first] += 1;
    counts[it->second] += 1;
  }
  for (int i = 0; i < size(); i++)
    maxCount = maxCount < counts[i] ? counts[i] : maxCount;

  delete[] counts;
}

//Look-up Operations

template <typename T>
int GraphStructure<T>::dimension()
{
  return (int)X.size();
}

template <typename T>
int GraphStructure<T>::size()
{
  if (X.size() > 0)
  {
    return (int)X[0].size();
  }
  return 0;
}

template <typename T>
void GraphStructure<T>::get_x(int i, T *xi)
{
  for (int d = 0; d < dimension(); d++)
    xi[d] = X[d][i];
}

template <typename T>
T GraphStructure<T>::get_x(int i, int j)
{
  return X[i][j];
}

template <typename T>
T GraphStructure<T>::min(int dim)
{
  T minX = X[dim][0];
  for (int i = 1; i < size(); i++)
    minX = minX > X[dim][i] ? X[dim][i] : minX;
  return minX;
}

template <typename T>
T GraphStructure<T>::max(int dim)
{
  T maxX = X[dim][0];
  for (int i = 1; i < size(); i++)
    maxX = maxX < X[dim][i] ? X[dim][i] : maxX;
  return maxX;
}

template <typename T>
T GraphStructure<T>::range(int dim)
{
  return max(dim) - min(dim);
}

template <typename T>
std::set<int> GraphStructure<T>::get_neighbors(int index)
{
  return neighbors[index];
}

template <typename T>
std::map<int, std::set<int>> GraphStructure<T>::full_graph()
{
  return neighbors;
}

template class GraphStructure<double>;
template class GraphStructure<float>;
