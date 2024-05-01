from Game_of_life import GameOfLife as GOL

import numpy as np
import pandas as pd

print(GOL.measure_equilibrium_time())

# number of simulations to account for in the histogram
sim_number = 1000

times = np.zeros(sim_number)

# iterate over a specified number of simulations and generate the equilibrium time
for i in range(sim_number):
    times[i] = GOL.measure_equilibrium_time()
    print(times[i])

# output equilibration times to datafile
df = pd.DataFrame(times)
df.to_csv("Checkpoint_2/Game_of_life/equilibration_times.csv", sep = ';', index = False)
