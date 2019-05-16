#%%
import numpy as np
import networkx as nx
import pandas as pd 
import json
import math
from scipy.spatial.distance import pdist, squareform

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
df = df.drop(columns=['Start date', 'End date', 'Start station', 'End station', 'Bike number', 'Member type']) #, 'Start station number', 'End station number', 'Bike number'])
df.head()
# df['Score'] = df.apply(weigthed_score, axis=1)

#%%
weigthed_score = lambda r: r[4]*0.5 + r[0]*0.15 + r[3]*0.15 + np.abs(r[1] - r[2])*0.2
weigthed_dist = lambda x,y: (weigthed_score(x) - weigthed_score(y))**2

#%%
dist = pdist(df.sample(100), metric=weigthed_dist)
dist = squareform(dist)
dist[dist == 0] = np.inf
edges = [(i, np.argmin(dist[:,i])) for i in range(len(dist))]


#%%
G = nx.Graph()
G.add_edges_from(edges)

nx.draw(G, pos=nx.spring_layout(G), alpha=0.6, node_size=50)


#%%
nx.write_gexf(G, "resources/bikes_100.gexf")

#%% [markdown]
### nx.spring_layout() --> Fruchterman Reingold Algorithm
#---
#### 1) Forces between nodes:
# * Attractive: spring force
# * Repulsive: electrical force
#### 2) Move nodes to minimze energy of the system
# * Iterations controlled by 'temp' (sim-annel)
#---
####  https://github.com/gephi/gephi/wiki/Fruchterman-Reingold
