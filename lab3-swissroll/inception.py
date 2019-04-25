#%%
%matplotlib inline
import math
from itertools import product
from math import pi
from time import time

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib.ticker import NullFormatter
from sklearn import datasets, manifold
import seaborn as sns

#%%
lle = np.load('lab3-swissroll/maps_lle.npy')
iso = np.load('lab3-swissroll/maps_iso.npy')
mds = np.load('lab3-swissroll/maps_mds.npy')
tsne = np.load('lab3-swissroll/maps_tsne.npy')

maps = np.concatenate((lle, iso, mds, tsne))

#%%
def paths():
    base = 'resources/images/'
    for (k,v) in (m for m in maps):
        Y, color, mtime = v
        title = str(k) + mtime
        yield base + title + '.png'

images_org = (plt.imread(path) for path in paths())
images = map(lambda x: x.reshape(5184,4), images_org)

combined = []
for img in images:
    combined.append(img)
combined = np.array(combined).reshape(48, 5184*4)

images_org = (plt.imread(path) for path in paths())
image_dict = {
    i: p
    for i, p in enumerate(images_org)
}

#%%
model = manifold.LocallyLinearEmbedding(n_jobs=-1)
Y = model.fit_transform(combined)

#%%
Z = np.array([i for i,path in enumerate(paths())])
Z = Z.reshape(48,1)
Z = np.concatenate((Y,Z), axis=1)
Z = sorted(Z, key=lambda x: (x[0], x[1]))
order = [int(z[2]) for z in Z]

#%%
fig, axes = plt.subplots(nrows=16, ncols=3, figsize=(3, 16))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for (o, ax) in zip(order, axes.flatten()):
    ax.imshow(image_dict[o])
    ax.plot()

