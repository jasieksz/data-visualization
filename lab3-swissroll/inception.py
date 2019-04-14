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

#%%
lle = np.load('lab3-swissroll/maps_lle.npy')
iso = np.load('lab3-swissroll/maps_iso.npy')
mds = np.load('lab3-swissroll/maps_mds.npy')
maps = np.concatenate((lle, iso, mds))

#%%
maps[0][1][1].shape

#%%
x = [(maps[i][0], maps[i][1][0].ravel()) for i in range(len(maps))]
x = {x[i][0]: list(x[i][1]) for i in range(len(x))}
y = np.array([np.array(v) for (k, v) in x.items()])

#%%
V = manifold.LocallyLinearEmbedding(10, 2, n_jobs=-1).fit_transform(y)
   
#%%
[(maps[i][0], maps[i][1][0].shape) for i in range(len(maps))]
