import numpy as np
import sklearn.neighbors
import time
import numba
import argparse
import sys

parser = argparse.ArgumentParser(description='Build an lp-beta skeleton using numba.')
parser.add_argument('-i', dest='filename', type=str,
                    help='The input data file as a csv.')

args = parser.parse_args()

start = time.time()
X = np.loadtxt(args.filename)
problem_size = X.shape[0]
dimensionality = X.shape[1]
end = time.time()
print('Load data ({} s) shape={}'.format(end-start, X.shape), file=sys.stderr)

if dimensionality < 5:
    kmax = 100
else:
    kmax = 200

max_neighbors = min(problem_size-1, kmax)

start = time.time()
@numba.jit(nopython=True)
def paired_lpnorms(A, B, p=2):
    """ Method to compute the paired Lp-norms between two sets of points. Note,
    A and B should be the same shape.

    Args:
        A (MxN matrix): A collection of points
        B (MxN matrix): A collection of points
        p (positive float): The p value specifying what kind of Lp-norm to use
            to compute the shape of the lunes.
    """
    N = A.shape[0]
    dimensionality = A.shape[1]
    norms = np.zeros(N)
    for i in numba.prange(N):
        norm = 0.0
        for k in range(dimensionality):
            norm += (A[i, k] - B[i, k])**p
        norms[i] = norm**(1./p)
    return norms

@numba.jit(nopython=True)
def min_distance_from_edge(t, beta, p):
    """ Using a parameterized scale from [0,1], this function will determine the
    minimum valid distance to an edge given a specified lune shape defined
    by a beta parameter for defining the radius and a p parameter specifying
    the type of Lp-norm the lune's shape will be defined in.
    
    Args:
        t (float): the parameter value defining how far into the edge we are.
        0 means we are at one of the endpoints, 1 means we are at the edge's
        midpoint.
        beta (float): The beta value for the lune-based beta-skeleton
        p (float): The p value specifying which Lp-norm to use to compute
            the shape of the lunes. A negative value will be
            used to represent the inf norm

    """
    xC = 0
    yC = 0
    if t > 1:
        return 0
    if beta <= 1:
        r = 1 / beta
        yC = (r**p - 1)**(1. / p)
    else:
        r = beta
        xC = 1 - beta
    y = (r**p - (t-xC)**p)**(1. / p) - yC
    return 0.5*y

@numba.jit(nopython=True,parallel=True,nogil=True)
def create_template(beta, p=2, steps=100):
    """ Method for creating a template that can be mapped to each edge in
    a graph, since the template is symmetric, it will map from one endpoint
    to the center of the edge.

    Args:
        beta (float [0,1]): The beta value for the lune-based beta-skeleton
        p (positive float): The p value specifying which Lp-norm to use to
            compute the shape of the lunes.
    """
    template = np.zeros(steps+1)
    if p < 0:
        if beta >= 1:
            template[:-1] = beta/2
        return template
    for i in numba.prange(steps):
        template[i] = min_distance_from_edge(i/steps, beta, p)
    return template

paired_lpnorms(X[:10], X[:10], 2)
min_distance_from_edge(0, 1, 2)
template = create_template(1, 2, 49)
end = time.time()
print('Precompile local functions ({} s)'.format(end-start), file=sys.stderr)
print('Compute preliminary neighborhood graph:', file=sys.stderr)

start = time.time()
knnAlgorithm = sklearn.neighbors.NearestNeighbors(max_neighbors)
knnAlgorithm.fit(X)
edges = knnAlgorithm.kneighbors(X, return_distance=False)
end = time.time()
print('\t SKL Default ({} s)'.format(end-start), file=sys.stderr)

# @numba.njit(numba.float64[:, ::1](numba.float64[:, ::1], numba.int64[:, ::1], numba.float64, numba.float64, numba.int64), parallel=True, fastmath=True, nogil=True)
@numba.njit()
def prune(X, edges, beta = 1, lp = 2, steps = 99):
    # problem_size = min(10000, edges.shape[0]) # edges.shape[0]
    problem_size = edges.shape[0]
    template = create_template(beta, lp, steps)
    pruned_edges = np.zeros(shape=edges.shape) - 1
    # timings = 18*[0]
    for i in numba.prange(problem_size):
        # print(i,problem_size)
        p = X[i]
        # Xp = X - p
        for k in range(edges.shape[1]):
            ###################################################################
            #  0	2%
            # start = time.time()
            j = edges[i, k]
            q = X[j]
            if i == j:
                continue
            # timings[0] += time.time() - start
            ###################################################################
            #  1	1%
            # start = time.time()
            pq = q - p
            # timings[1] += time.time() - start
            ###################################################################
            #  2	8%
            # start = time.time()
            edge_length = np.linalg.norm(pq)
            # timings[2] += time.time() - start
            ###################################################################
            #  3	4%
            # start = time.time()
            # adjacent_indices = []
            # for m in range(len(edges)):
            #     row = edges[m]
            #     if np.any(np.logical_or(row == i, row == j)):
            #         adjacent_indices.append(m)
            # subset = np.concatenate((edges[i], edges[j], np.array(adjacent_indices)))
            subset = np.concatenate((edges[i], edges[j]))
            # timings[3] += time.time() - start
            ###################################################################
            #  4	21%
            # start = time.time()
            # subset = np.unique(subset)
            # timings[4] += time.time() - start
            ###################################################################
            #  5	11%
            # start = time.time()
            Xp = X[subset] - p
            # timings[5] += time.time() - start
            ###################################################################
            #  6	8%
            # start = time.time()
            projections = np.dot(Xp, pq)/(edge_length**2)
            # timings[6] += time.time() - start
            ###################################################################
            #  7	5%
            # start = time.time()
            temp_indices_1 = projections * 2 - 1
            # timings[7] += time.time() - start
            ###################################################################
            #  8	2%
            # start = time.time()
            temp_indices_2 = steps*temp_indices_1
            # timings[8] += time.time() - start
            ###################################################################
            #  9	2%
            # start = time.time()
            temp_indices_3 = np.rint(temp_indices_2)
            # timings[9] += time.time() - start
            ###################################################################
            # 10	1%
            # start = time.time()
            temp_indices_4 = temp_indices_3.astype(np.int64)
            # timings[10] += time.time() - start
            ###################################################################
            # 11	2%
            # start = time.time()
            lookup_indices = np.abs(temp_indices_4)
            # timings[11] += time.time() - start
            ###################################################################
            # 12	3%
            # start = time.time()
            temp_indices_5 = np.logical_and(lookup_indices >= 0, lookup_indices <= steps)
            # timings[12] += time.time() - start
            ###################################################################
            # 13	3%
            # start = time.time()
            valid_indices = np.nonzero(temp_indices_5)[0]
            # timings[13] += time.time() - start
            ###################################################################
            # 14	9%
            # start = time.time()
            temp = np.atleast_2d(projections[valid_indices]).T*pq
            # timings[14] += time.time() - start
            ###################################################################
            # 15	8%
            # start = time.time()
            distances_to_edge = paired_lpnorms(Xp[valid_indices], temp)
            # timings[15] += time.time() - start
            ###################################################################
            # 16	7%
            # start = time.time()
            points_in_region = np.nonzero(distances_to_edge < edge_length*template[lookup_indices[valid_indices]])[0]
            # timings[16] += time.time() - start
            ###################################################################
            # 17	1%
            # start = time.time()
            if len(points_in_region) == 0:
                pruned_edges[i, k] = j
            # timings[17] += time.time() - start
    # return pruned_edges, timings
    return pruned_edges

start = time.time()
knnAlgorithm = sklearn.neighbors.NearestNeighbors(100)
knnAlgorithm.fit(X[:100])
small_edge_list = knnAlgorithm.kneighbors(X, return_distance=False)
prune(X[:100], small_edge_list, beta = 1., lp = 2., steps = 49)
end = time.time()
print('Precompile prune function ({} s)'.format(end-start), file=sys.stderr)

start = time.time()
timings = None
# pruned_edges, timings = prune(X, edges, beta = 1, lp = 2, steps = 49)
pruned_edges = prune(X, edges, beta = 1, lp = 2, steps = 999)
end = time.time()
print('Actual prune function ({} s)'.format(end-start), file=sys.stderr)

# if timings is not None:
#     for i,t in enumerate(timings):
#         print(i, t)

# outfile = open('../edges_{}D_numba.txt'.format(dimensionality), 'w')
for i,p in enumerate(pruned_edges):
    for q in p:
        lo, hi = (int(i), int(q)) if i < q else (int(q), int(i))
        if q != -1 and q != i:
            print('{} {}\n'.format(lo,hi))
# outfile.close()

# edge_list = set()
# for i,p in enumerate(edges):
#     for q in p:
#         lo, hi = (int(i), int(q)) if i < q else (int(q), int(i))
#         if q != -1 and q != i:
#             edge_list.add((lo, hi))

# print('KNN: {}'.format(len(edge_list)))