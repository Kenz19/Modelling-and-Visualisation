'''
Plot sliced diagram
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

# read in data
# Is = pd.read_csv('Checkpoint_2/SIRS/Plots/Cut/phase.csv', header = None)
# Is2 = pd.read_csv('Checkpoint_2/SIRS/Plots/Cut/phase2.csv', header = None)
# IsErr = pd.read_csv('Checkpoint_2/SIRS/Plots/Cut/phase_err.csv', header = None)

Is = pd.read_csv('Checkpoint_2/SIRS/phase.csv', header = None)
Is2 = pd.read_csv('Checkpoint_2/SIRS/phase2.csv', header = None)
IsErr = pd.read_csv('Checkpoint_2/SIRS/phase_err.csv', header = None)

# get in form of array
I = np.array(Is)[:, 0]
I2 = np.array(Is2)[:, 0]
IErr = np.array(IsErr)[:, 0]

p1_vals = np.arange(0.2, 0.5 + 0.02, 0.02)
print(len(p1_vals))

print(I, I2, IErr)

y = (I2 - I**2)/2500
print(y)

plt.errorbar(p1_vals, y, IErr)
plt.ylabel('$(\langle I^2 \\rangle - \langle I \\rangle^2)/N$', fontsize = 20)
plt.xlabel('$p_1$', fontsize = 20)
plt.title('Cut along phase diagram where $p_3$ = 0.5', fontsize = 20)
plt.show()

dict = {'p1': p1_vals, '<I>': I, '<I2>': I2, 'var': y, 'Ierr': IErr}

data = pd.DataFrame.from_dict(dict, orient='columns')#, dtype=None, columns=None)

# output the average number of infected cells
data.to_csv("cut_data.csv", index = False)