import numpy as np
from sklearn import neighbors
import samply


def yao_graph(X, D, k, max_neighbors=None):
    if max_neighbors is None:
        max_neighbors = len(X)
    vectors = samply.SCVTSampler.generate_samples(k, D)
    edges = []

    nn = neighbors.NearestNeighbors(n_neighbors=max_neighbors)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    # Do not include yourself in the query results
    if np.max(distances[:, 0]) < 1e-15:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        max_neighbors -= 1

    for i in range(X.shape[0]):
        projections = np.dot(X[indices[i]] - X[i], vectors.T)
        representatives = np.argmax(projections, axis=1)
        used = np.zeros(len(vectors), dtype=bool)
        j = 0
        while not np.all(used) and j < max_neighbors:
            rep = representatives[j]
            if not used[rep]:
                index = indices[i, j]
                if index < i:
                    edges.append((index, i, distances[i, j]))
                else:
                    edges.append((i, index, distances[i, j]))
                used[rep] = True
            j += 1
    return edges


def theta_graph(X, D, k, max_neighbors=None):
    if max_neighbors is None:
        max_neighbors = len(X)
    vectors = samply.SCVTSampler.generate_samples(k, D)
    edges = []

    nn = neighbors.NearestNeighbors(n_neighbors=max_neighbors)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    # Do not include yourself in the query results
    if np.max(distances[:, 0]) < 1e-15:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        max_neighbors -= 1

    for i in range(X.shape[0]):
        projections = np.dot(X[indices[i]] - X[i], vectors.T)
        representatives = np.argmax(projections, axis=1)
        max_projections = np.max(projections, axis=1)
        used = np.zeros(len(vectors), dtype=bool)
        order = reversed(np.argsort(max_projections))
        for j in order:
            rep = representatives[j]
            if not used[rep]:
                index = indices[i, j]
                if index < i:
                    edges.append((index, i, distances[i, j]))
                else:
                    edges.append((i, index, distances[i, j]))
                used[rep] = True
                if np.all(used):
                    break
    return edges
