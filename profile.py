import nglpy
import sklearn.neighbors
import time
import numpy as np

# import matplotlib.pyplot as plt

# plt.axis()
# plt.ion()
# plt.show()
# plt.gca().set_xlim(0, 30)

# colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3',
#           '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']
# glyphs = ['^', 's', 'p', 'h', '8', 'o']

print('Dimension Samples   K Seed Sklearn_time NGLpy_time Bskeleton_time')
for d in range(9, 30):
    for n in range(6, 7):
        N = pow(10, n)
        for seed in range(1):
            np.random.seed(seed)
            X = np.random.uniform(0, 1, (N, d))
            np.savetxt('data_{}_{}_{}.csv'.format(d, N, seed), X, delimiter=',')

            for i, k in enumerate([200]):
                if k > N-1:
                    k = N-1

                start = time.time()
                knnAlgorithm = sklearn.neighbors.NearestNeighbors(k)
                knnAlgorithm.fit(X)
                edges = knnAlgorithm.kneighbors(X, return_distance=False)
                end = time.time()
                time_knn = end-start

                # start = time.time()
                # graph_rep = nglpy.Graph(X, 'approximate knn', k, 1)
                # end = time.time()
                time_ngl = end-start

                # start = time.time()
                # graph_rep = nglpy.Graph(X, 'beta skeleton', k, 1)
                # end = time.time()
                time_bs = end-start

                print('{:>9} {:>7} {:>3} {:>4} {:>12.6f} {:>10.6f} {:>14.6f}'
                      .format(d, N, k, seed, time_knn, time_ngl, time_bs))

            # avg_knn = np.average(times_knn)
            # avg_ngl = np.average(times_ngl)
            # print('~'*80)
            # print(d, N, k, avg_knn, avg_ngl)
            # print('~'*80)

            # plt.scatter(d, avg_knn, c=colors[i], marker=glyphs[n-1])
            # plt.scatter(d, avg_ngl, c=colors[len(colors)-1-i], marker=glyphs[n-1])

            # plt.gca().set_ylim(0, max(np.max(times_knn), np.max(times_ngl)))
            # plt.pause(0.05)

# plt.show()