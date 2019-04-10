#%%
%matplotlib inline
# %matplotlib notebook

import math
from itertools import product
from math import pi
from time import time

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib.ticker import NullFormatter
from sklearn import datasets, manifold

#%% Generators
def make_spiral(n, big_r, alpha, start_r):
    r = start_r
    points = []
    for x in range(0, n):
        fig_r = math.cos(4 * pi / n * x) * r
        a = np.array([(big_r + fig_r) * math.cos(alpha), (big_r + fig_r) * math.sin(alpha), math.sin(4 * pi / n * x) * r, x])
        points.append(a)
        r = start_r * (n - x) / n

    points = np.array(points)
    noise_matrix = np.random.normal(0, .1, points.shape)
    noise_matrix = noise_matrix.reshape(points.shape)
    result = np.add(points, noise_matrix)
    return result

def make_roll(rn, rr, sn, sr, sam=1): # no. spirals, donugth R, no. points in spiral, spiral R, dist between spirals
    alphas = [sam * pi/2 * x / rn for x in range(0, rn + 1)]
    roll = make_spiral(sn, rr, alphas[0], sr)
    for p in alphas[1:]:
        roll = np.concatenate((roll, make_spiral(sn, rr, p, sr)))
    return (roll[:,:3], roll[:,3])

#%% Generate roll variants
spiral_points = [50, 150, 450]
interspaces = [0.5, 1, 2]
n_spirals = 20
dounugth_r = 25
spiral_r = 10

rolls = {
    (sp, insp): make_roll(n_spirals, dounugth_r, sp, spiral_r, insp)
    for (sp, insp) in product(spiral_points, interspaces)
}

#%% Show roll variants
fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
for i, (k, v) in enumerate(rolls.items()):
    X, color = v
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.set_title(str(k))
plt.show()

#%% Generate models
components = 2
neighbors = 8

models = {
    # ('tsne', 0) : manifold.TSNE(components, init='pca', random_state=0),
    ('mds', 0) : manifold.MDS(components, max_iter=100, n_init=1),
    ('lle', neighbors) : manifold.LocallyLinearEmbedding(neighbors, components),
    ('lle', neighbors*2) : manifold.LocallyLinearEmbedding(neighbors*2, components),
    ('isomap', neighbors) : manifold.Isomap(neighbors, components),
    ('isomap', neighbors*2) : manifold.Isomap(neighbors*2, components)
}

#%% Generate mappings
def get_mapping(rolls):
    for (name, model) in models.items():
        for (params, roll) in rolls.items():
            X, color = roll
            t0 = time()
            Y = model.fit_transform(X)
            t1 = time()
            yield (name[0] + ' ' + str(name[1]) + ' ' + str(params)), (Y, color, ' time:' + "{:.2f}".format(t1-t0))

def get_model_mapping(maps):
    for s in range(0, len(maps), 9):
        yield maps[s:s+9]

maps = np.array(list(get_mapping(rolls)))

#%% Show mappings
for mapp in get_model_mapping(maps):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    for ax, mapp in zip(axes.flatten(), mapp):
        k, v = mapp
        Y, color, mtime = v
        ax.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
        ax.set_title(str(k) + mtime)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

