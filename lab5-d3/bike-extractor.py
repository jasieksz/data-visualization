#%%
import numpy as np
import pandas as pd
import sys 
from matplotlib import pyplot as plt
from collections import Counter 
import random
from matplotlib import cm
from matplotlib import colors

#%%
path = '../resources/201806-capitalbikeshare-tripdata.csv'
df = pd.read_csv(path)
df.head()

#%%
stations = list(df['Start station number'].unique())
stations.append(31819) # Missing station number from 'End station number'
stations = sorted(stations)
MIN = min(stations)
MAX = max(stations)

#%% CSV : station,color
color = [colors.to_hex(e) for e in 
        [(r,g,b) for r,g,b,a in [cm.jet(i) for i in np.linspace(0, 1.0, len(stations))]]]
# color = np.linspace(0, 1.0, len(stations))
print('name,color')
for s,c in zip(stations, color):
    print(str(s) + ',' + str(c))

#%% Connection matrix
connections = {(k1,k2): 0 for k1 in stations for k2 in stations}
for index, value in df[['Start station number', 'End station number']].iterrows():
    connections[(value[0], value[1])] = connections.get((value[0], value[1]), 0) + 1

connectionMat = np.zeros((MAX-MIN+1, MAX-MIN+1), dtype=int)
for k,v in connections.items():
    a,b = k
    connectionMat[a-MIN][b-MIN] = v

print(np.array2string(connectionMat, threshold=sys.maxsize, separator= ', '))
