from Game_of_life import GameOfLife as GOL

import numpy as np
import pandas as pd

print('The centre of mass velocity of the glider is ' + str(GOL.measure_centre_of_mass_velocity()) + ' cells/iteration')