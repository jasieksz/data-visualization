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
sns.set(style="darkgrid")

#%%
path = 'resources/201806-capitalbikeshare-tripdata.csv'
df = pd.read_csv(path)
df.head()

#%%
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

stations.describe()

#%%
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


#%%
id_coord = stations.filter(['short_name','lat','lon'], axis=1)
id_cord = id_coord.to_dict(orient='records')

#%%

def s_gen():
    for i, row in id_coord.iterrows():
        yield row['short_name'], (row['lat'], row['lon'])

stats = {
    k: v
    for (k, v) in s_gen()
}

#%%
stats

#%%
def map_type(member):
    if member == 'Member': 
        return 10
    return 0

# df['StationDelta'] = df['End station number'] - df['Start station number']
# # df['Member type'] = df['Member type'].transform(lambda t: map_type(t))
# # df['Bike number'] = df['Bike number'].transform(lambda n: re.sub(r'[^0-9]', '', n))
# # df = df.drop(columns=['Start date', 'End date', 'Start station', 'End station'])
# df.head()


#%%
model = manifold.LocallyLinearEmbedding(10, 3, n_jobs=-1)
Z = model.fit_transform(df.head(1000))
Z = np.hstack((Z, df.head(1000)['Member type'].as_matrix().reshape(1000,1)))
sns.scatterplot(Z[:,1], Z[:,0], size=Z[:,2], hue=Z[:,3], legend=None)


#%%
sns.scatterplot(x='Start station number',
    y='End station number',
    hue='Member type',
    data=df.head(10000),
    alpha=0.8)



#%%
tum = pd.read_csv('tumor_10000.data', sep='\s+', header=None)

#%%
tum.describe()
