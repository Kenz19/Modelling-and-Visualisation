'''
Plot phase diagram of the variance of the number of infected sites
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read in I
Is = pd.read_csv('Checkpoint_2/SIRS/Plots/Phase_diagram/Sweeps=1000/phase.csv')
I = np.array(Is)**2
print(I)

Is2 = pd.read_csv('Checkpoint_2/SIRS/Plots/Phase_diagram/Sweeps=1000/phase2.csv')
I2 = np.array(Is2)
print(I2)

var = (I2 - I)/2500
print(var)


plt.imshow(var, origin = 'lower', extent =  [0,1, 0,1])
plt.xlabel('$p_3$')
plt.ylabel('$p_1$')
plt.title('$p_2$ = 0.5')
plt.colorbar(label = '$\\frac{\langle I^2 \\rangle - \langle I \\rangle^2}{N}$')
plt.show()