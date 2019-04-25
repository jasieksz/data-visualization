#%%
# %matplotlib inline
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
spiral_points = [100]#, 150, 450]
interspaces = [1]
n_spirals = 15
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
    # ('tsne', 10) : manifold.TSNE(components, init='pca', random_state=0, n_iter=1000, perplexity=10, learning_rate=100),
    # ('tsne', 50) : manifold.TSNE(components, init='pca', random_state=0, n_iter=1000, perplexity=50, learning_rate=100),
    # ('tsne', 100) : manifold.TSNE(components, init='pca', random_state=0, n_iter=1000, perplexity=100, learning_rate=100),
    # ('tsne-i', 5) : manifold.TSNE(components, init='pca', random_state=0, n_iter=500, perplexity=50, learning_rate=100),
    # ('tsne-i', 10) : manifold.TSNE(components, init='pca', random_state=0, n_iter=1000, perplexity=50, learning_rate=100),
    # ('tsne-i', 50) : manifold.TSNE(components, init='pca', random_state=0, n_iter=5000, perplexity=50, learning_rate=100),
    # ('mds', 0) : manifold.MDS(components, max_iter=100, n_init=1, n_jobs=-1),
    # ('lle', neighbors) : manifold.LocallyLinearEmbedding(neighbors, components, n_jobs=-1),
    # ('lle', neighbors*2) : manifold.LocallyLinearEmbedding(neighbors*2, components, n_jobs=-1),
    # ('isomap', neighbors) : manifold.Isomap(neighbors, components, n_jobs=-1),
    # ('isomap', neighbors*2) : manifold.Isomap(neighbors*2, components, n_jobs=-1)
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

def get_model_mapping(arr):
    for s in range(0, len(arr), 9):
        yield arr[s:s+9]

maps = np.array(list(get_mapping(rolls)))
np.save('maps_tsne_i.npy', maps)

#%% Load saved mappings
lle = np.load('lab3-swissroll/maps_lle.npy')
iso = np.load('lab3-swissroll/maps_iso.npy')
mds = np.load('lab3-swissroll/maps_mds.npy')
tsne = np.load('lab3-swissroll/maps_tsne.npy')
maps = np.concatenate((lle, iso, tsne, mds))


#%% Show mappings
for mapp in get_model_mapping(maps):
    fig, axes = plt.subplots(nrows=mapp.shape[0]//3, ncols=3, figsize=(15,mapp.shape[0]//3*5))
    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    for ax, mapp in zip(axes.flatten(), mapp):
        k, v = mapp
        Y, color, mtime = v
        ax.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
        ax.set_title(str(k) + mtime)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

#%%
for (k,v) in (m for m in maps):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1, 1))
    Y, color, mtime = v
    ax.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
    title = str(k) + mtime
    # ax.set_title(title)
    fig.savefig('resources/images/' + title + '.png')
    plt.close(fig)
