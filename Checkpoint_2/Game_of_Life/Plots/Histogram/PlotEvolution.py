import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plotting style
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'roman',})

#handle graph formatting and style
plt.style.use('shendrukGroupStyle')

# read in the equilibration times file
#data = pd.read_csv('Checkpoint_2/Game_of_life/equilibration_times.csv')#
data = pd.read_csv('equilibration_times.csv')
times = np.array(data)[:, 0]

print(times)

# remove any nans
times = times[~np.isnan(times)]
print(times)

# plot 
fig, ax = plt.subplots(nrows = 1, ncols = 1)

# needs to be a percentage on the y axis
ax.hist(times, bins = 50)
ax.set_xlabel('Equilibration time (iterations)')
ax.set_ylabel('Count')

plt.show()