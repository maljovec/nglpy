import numpy as np

f32 = np.float32
i32 = np.int32


class Graph(object):
    """A neighborhood graph inducing a 1-skeleton on a arbitrary dimensional
    point cloud using one of the empty region graphs from the ngl library.

    Attributes:
        None
    """

    def __init__(
        self,
        X: list[float],
        rows: int,
        cols: int,
        graph: str,
        maxN: int,
        beta: float,
        edgeIndices: list[int],
    ):
        self.X = X
        self.rows = rows
        self.cols = cols
        self.graph = graph
        self.maxN = maxN
        self.beta = beta
        self.edgeIndices = edgeIndices

    def dimension(self):
        return self.cols

    def size(self):
        return self.rows

    def max(self, dim: int):
        return max(v for v in self.X[dim :: self.cols])

    def min(self, dim: int):
        return min(v for v in self.X[dim :: self.cols])

    def range(self, dim: int):
        return self.max(dim) - self.min(dim)

    def get_x(self, row: int, col: int = None):
        if col is not None:
            return self.X[row * self.cols + col]
        start_index = row * self.cols
        return self.X[start_index : (start_index + self.cols)]

    def full_graph(self):
        pass

    def get_neighbors(self, index: int):
        pass
