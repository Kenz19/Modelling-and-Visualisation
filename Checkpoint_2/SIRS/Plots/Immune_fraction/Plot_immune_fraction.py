'''
Plotting script for the fraction of infected cells vs fraction of immune cells
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plotting style
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'roman',})

#handle graph formatting and style
plt.style.use('shendrukGroupStyle')


# read in data file
data = pd.read_csv('infected_vs_fraction.csv')
arr = np.array(data)

frac = arr[:, 0]
I = arr[:, 1]

# divide by N (2500)
print(data)

plt.scatter(frac, I)
plt.xlabel('$f_{im}$')
plt.ylabel('$\langle I \\rangle/N$')

plt.show()