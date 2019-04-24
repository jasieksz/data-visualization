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

images = (plt.imread(path) for path in paths())
images = map(lambda x: x.reshape(5184,4), images)

combined = []
for img in images:
    combined.append(img)

# combined = np.vstack([img for img in images], axis=0)
combined = np.array(combined).reshape(48, 5184*4)

#%%
model = manifold.LocallyLinearEmbedding(n_jobs=-1)
Y = model.fit_transform(combined)

#%%
Z = np.array([i for i,path in enumerate(paths())])
Z = Z.reshape(48,1)
Z = np.concatenate((Y,Z), axis=1)
Z = sorted(Z, key=lambda x: (x[0], x[1]))

#%%
order = [int(z[2]) for z in Z]
