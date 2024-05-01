import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read in phase.csv

data = pd.read_csv('Checkpoint_2/SIRS/phase.csv')
I = np.array(data)/2500

resolution = 0.05

tick_vals = np.arange(0, 1 + resolution, resolution)

print(np.max(I), np.min(I))

plt.imshow(I, origin = 'lower', extent =  [0,1, 0,1])
plt.xlabel('$p_3$')
plt.ylabel('$p_1$')
plt.title('$p_2$ = 0.5')
plt.colorbar(label = '$\langle I \\rangle/N$')
plt.show()

#plt.savefig('Checkpoint_2/SIRS/Phase_200.png')