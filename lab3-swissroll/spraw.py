#%%
from matplotlib import pyplot as plt
import os

#%%
base = 'resources/images/'
rolls = base + 'rolls.png'
maps = (base + 'roll_' + str(i) + '.png' for i in range(6))
inception = [base + 'inception.png', base + 'inception_2.png']
bikes = (base + 'bikes_' + m + '.png' for m in ['lle', 'iso', 'tsne'])
art = sorted(os.listdir(base + 'results/'))

#%%


#%%
display_image_in_actual_size(rolls)