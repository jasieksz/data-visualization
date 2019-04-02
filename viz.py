#%%
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from collections import Counter 
import seaborn as sns
sns.set(style="darkgrid")

#%%
path = '201806-capitalbikeshare-tripdata.csv'
df = pd.read_csv(path)
df.head()

#%%
time_by_bike = df.groupby('Bike number')['Duration'].mean()
usage_by_bike = df.groupby('Bike number')['Bike number'].count().rename('Count')
bikes = sorted(list(df['Bike number'].unique()))

#%%
usage_time_by_bike = pd.concat([usage_by_bike, time_by_bike], axis=1)
usage_time_by_bike.head()

#%%
count = list(usage_time_by_bike['Count'])
duration = list(usage_time_by_bike['Duration'])
b,c,d = zip(*sorted(list(zip(bikes, count, duration)), key=lambda x: (x[1], x[2], x[0]), reverse=True))

#%%
data = pd.DataFrame({
    'count': c[:1000],
    'duration': d[:1000]
})

g = sns.JointGrid(x="duration", y="count", data=data)
g = g.plot_joint(plt.scatter, color=".5", edgecolor="white")
g = g.plot_marginals(sns.distplot, kde=False, color=".5")

#%%
g = sns.JointGrid(x="duration", y="count", data=data, space=0)
g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
g = g.plot_marginals(sns.kdeplot, shade=True)