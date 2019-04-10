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