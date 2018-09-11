import faiss


class SearchIndex(object):
    def __init__(self, X):
        pass


class FaissSearchIndex(SearchIndex):
    def __init__(self, X, name):
        D = X.shape[1]
        self.index = faiss.index_factory(D, name)
        self.index.train(X)
        self.index.add(X)

    def search(self, xq, k):
        return self.index.search(xq, k)


class FaissGPUApproximateIndex(FaissSearchIndex):
    def __init__(self, X, name):
        D = X.shape[1]
        res = faiss.StandardGpuResources()
        index = faiss.index_factory(D, name)
        co = faiss.GpuClonerOptions()

        # here we are using a 64-byte PQ, so we must set the lookup
        # tables to 16 bit float (this is due to the limited temporary
        # memory).
        co.useFloat16 = True
        self.index = faiss.index_cpu_to_gpu(res, 0, index, co)
        self.index.train(X)
        self.index.add(X)


class FaissGPUExactIndex(FaissSearchIndex):
    def __init__(self):
        D = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        self.index = faiss.GpuIndexFlatL2(res, D, flat_config)
        self.index.train(X)
        self.index.add(X)


class FaissExactIndex(FaissSearchIndex):
    def __init__(self, X):
        D = X.shape[1]
        self.index = faiss.IndexFlatL2(D)
        self.index.train(X)
        self.index.add(X)
