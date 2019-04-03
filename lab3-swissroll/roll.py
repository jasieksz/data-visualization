
#%%
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
from time import time
from matplotlib.ticker import NullFormatter


#%%
X, color = datasets.samples_generator.make_swiss_roll(n_samples=1500)

print("Computing LLE embedding")
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
print("Done. Reconstruction error: %g" % err)


#%%
get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()

ax = fig.add_subplot(211, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")
ax = fig.add_subplot(212)
ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data')
plt.show()

#%%
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.datasets.samples_generator import make_swiss_roll
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import math
from math import pi
import numpy as np

#%%
def make_spiral(big_r, alpha, n, start_r):
    r = start_r
    points = []
    for x in range(0, n + 1):
        fig_r = math.cos(4 * pi / n * x) * r
        points.append(np.array([(big_r + fig_r) * math.cos(alpha), (big_r + fig_r) * math.sin(alpha), math.sin(4 * pi / n * x) * r, x]))
        r = start_r * (n + 1 - x) / n

    points = np.array(points)
    noise_matrix = np.random.normal(0, .1, points.shape)
    noise_matrix = noise_matrix.reshape(points.shape)
    result = np.add(points, noise_matrix)
    return result

def make_roll(n_roll, r_roll, n, r):
    alphas = [pi/2 * x / n_roll for x in range(0, n_roll + 1)]
    roll = make_spiral(r_roll, alphas[0], n, r)
    for p in alphas[1:]:
        roll = np.concatenate((roll, make_spiral(r_roll, p, n, r)))
    return roll

#%%
def color_roll():
    n_roll = 30
    r_roll = 15
    spiral_points = 100
    spiral_r = 5
    
    X = make_roll(n_roll, r_roll, spiral_points, spiral_r)
    return (X[:,:3], X[:,3])


#%%
X, color = color_roll()
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(20, -60)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=10, cmap=plt.cm.Spectral)
plt.show()

#%% [markdown]
# ## LLE

#%%
kX_r, kerr = manifold.locally_linear_embedding(kX, n_neighbors=12, n_components=2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(kX_r[:, 0], kX_r[:, 1], c=kColor, cmap=plt.cm.viridis)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data')
plt.show()

#%%
X, color = color_roll()

fig = plt.figure(figsize=(20, 12))


n_neighbors = 10
n_components = 2

methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

for i, method in enumerate(methods):
    t0 = time()
    Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                        eigen_solver='auto',
                                        method=method).fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (methods[i], t1 - t0))

    ax = fig.add_subplot(252 + i)
    plt.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
    plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(2, 5, 10)
plt.scatter(Y[:, 0], Y[:, 1], s=2, c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()