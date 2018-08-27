import nglpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

########################################################################
draw_edges = True
draw_circles = False
annotate = False
N = 1000
########################################################################

np.random.seed(0)
D = 2
X = np.random.uniform(size=(N, D)).astype('float32')
go = nglpy.Graph(X, graph='', max_neighbors=5, beta=1.0)

edges = set()
outfile = '{}_uniform_{}_2_5_0.txt'.format('python', N)
with open(outfile, 'w') as f:
    for i, value in go.edges.items():
        for j in value:
            lo = i
            hi = j
            if j < i:
                lo, hi = hi, lo
            edges.add((lo, hi))
            f.write('{} {}\n'.format(lo, hi))

fig, ax = plt.subplots()

color = '#FED950'

if draw_edges:
    lines = []
    for edge in edges:
        lines.append([(X[edge[0], 0], X[edge[0], 1]),
                      (X[edge[1], 0], X[edge[1], 1])])
    lc = mc.LineCollection(lines, colors=color,
                           linewidths=1, linestyles='--')
    ax.add_collection(lc)

if draw_circles:
    for edge in edges:
        lo = edge[0]
        hi = edge[1]
        mdpt = (X[hi] + X[lo])/2.
        diameter = np.linalg.norm(X[hi] - X[lo])
        radius = diameter/2.

        empty_region = plt.Circle(
            (mdpt[0], mdpt[1]), radius, color=color, alpha=0.5)
        ax.add_artist(empty_region)
        ax.plot(X[[lo, hi], 0], X[[lo, hi], 1],
                c=color, linewidth=2, label='test')

ax.scatter(X[:, 0], X[:, 1], s=1, c='#fa9fb5')
if annotate:
    for i in range(len(X)):
        ax.annotate(i, (X[i, 0], X[i, 1]))

ax.autoscale()
ax.margins(0.1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# plt.legend()
plt.show()
