#%%
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from collections import Counter 
import seaborn as sns
import re
from matplotlib.ticker import NullFormatter
from sklearn import datasets, manifold
import json
import math
sns.set(style="darkgrid")

#%% Extract station_id, lat, lon
stations = None
with open('resources/station_information.json', 'r') as f:
    stations = json.load(f)

stations = stations['data']['stations']
stations = pd.DataFrame.from_dict(stations)
stations = stations.drop(['eightd_has_key_dispenser',\
    'eightd_station_services',\
    'electric_bike_surcharge_waiver',\
    'external_id',\
    'rental_url'], axis=1)

id_coord = stations.filter(['short_name','lat','lon'], axis=1)

def s_gen():
    for i, row in id_coord.iterrows():
        yield row['short_name'], (row['lat'], row['lon'])

stats = {
    k: v
    for (k, v) in s_gen()
}

#%% Preprocess bike data
def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))

path = 'resources/201806-capitalbikeshare-tripdata.csv'
df = pd.read_csv(path)

df['Distance'] = df.get(['Start station number', 'End station number']) \
    .apply(lambda r: haversine(stats.get(str(r[0]), (0,0)), \
                             stats.get(str(r[1]), (0,0))), axis=1)

df['Speed'] = df['Distance'] / df['Duration']
df = df[df.Speed < 20] # filter anomallies 
df = df.drop(columns=['Start date', 'End date', 'Start station', 'End station', 'Bike number']) #, 'Start station number', 'End station number', 'Bike number'])
df.head()

#%%
weigthed_score = lambda r: r[4]*0.5 + r[0]*0.15 + r[3]*0.15 + np.abs(r[1] - r[2])*0.2
weigthed_dist = lambda x,y: (weigthed_score(x) - weigthed_score(y))**2

models = {
    'tsne' : manifold.TSNE(2, init='pca', random_state=0, n_iter=2000, perplexity=40, learning_rate=500, metric=weigthed_dist),
    'lle' : manifold.LocallyLinearEmbedding(n_neighbors=8, n_jobs=-1),
    'isomap' : manifold.Isomap(n_neighbors=8, n_jobs=-1),
}


#%%
n_rows = 10000
Z = models['isomap'].fit_transform(df.drop(columns=['Member type']).head(n_rows))
Z = np.hstack((Z, df.head(n_rows)['Member type'].as_matrix().reshape(n_rows,1)))
np.save('lab3-swissroll/iso_bikes.npy', Z)

#%%
def show_mappings(Z):
    sns.scatterplot(Z[:,0], Z[:,1], hue=Z[:,2])

TSNE = np.load('lab3-swissroll/tsne_bikes.npy')
LLE = np.load('lab3-swissroll/lle_bikes.npy')
ISO = np.load('lab3-swissroll/iso_bikes.npy')

#%%
show_mappings(TSNE) 

#%%
show_mappings(LLE)

#%%
show_mappings(ISO)
