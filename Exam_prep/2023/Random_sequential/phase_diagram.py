import numpy as np
import matplotlib.pyplot as plt
from RS import SIRS
from tqdm import tqdm


def phase_diagram(upperp2 = 0.3, lowerp2 = 0, upperp3 = 0.3, lowerp3 = 0, p1 = 0.5, resolution = 0.01, sweeps = 1000, no_equilibration_sweeps = 100, p3 = 'else'):
    '''

    Determines a phase diagram varied over p1 and p3, p2 is set to a constant.

    Parameters:
    ----------
    upperp1: float
        Max value of p1 to be included

    lowerp1: float
        Min value of p1 to be included

    upperp3: float
        Max value of p3 to be included

    lowerp3: float
        Min value of p3 to be included   

    p2: float
        Value of p2 to be included

    resolution: float
        step size between upper and lower values of p1 and p3

    sweeps: integer
        Number of sweeps to carry out in each simulation

    no_equilibration_sweeps: integer
        Number of sweeps to pass before measurements start

    p3: string
        Either 'const' or 'None'. const means that the p3 value will be held at 0.5 and not varied over

    '''

    # generate list of p2 and p3 values based on specified resolution
    p2_vals = np.arange(lowerp2, upperp2 + resolution, resolution)
    no_p1 = len(p2_vals)
    print(no_p1)
    p3_vals = np.arange(lowerp3, upperp3 + resolution, resolution)
    no_p3 = len(p3_vals)

    # where the minority fraction will be stored
    average_fraction = np.zeros([no_p1, no_p3])

    # loop over p1 values
    for i in tqdm(range(no_p1)):
        # loop over p3 values

        for j in tqdm(range(no_p3)):

            # initialise SIRS grid
            sirs = SIRS(p1 = p1, p2 = p2_vals[i], p3 = p3_vals[j], N1 = 10, N2 = 10, initialisation = 'random', type = 'else', sweeps = sweeps)

            #array for minority fraction for each iteration of the current p2 and p3 combination
            #frac = np.zeros(sweeps - no_equilibration_sweeps) 
       
            #average_fraction[i] = sirs.minority_state(p3[i], equilibrium_sweeps = 100, sweeps = 1000)[0]

            average_fraction[i, j] = sirs.minority_state(equilibrium_sweeps = 100, sweeps = 1000)[0]
            # begin simulation
            # for k in range(sweeps):
                
            #     # update the sirs grid
            #     sirs.evolve_infection()

            #     # if system has reached equilibrium
            #     if k >= no_equilibration_sweeps:

            #         frac[k-no_equilibration_sweeps] = (sirs.minority_state(equilibrium_sweeps = 100, sweeps = 1000)[0])/(sirs.rows*sirs.cols)


            # # output the mean infected measurement for this p1 and p3 combination
            # average_fraction[i, j] = np.mean(frac)

            #print(I, I2, Ierr)
    #print(I, I2, Ierr)

    # # output the average number of infected cells
    np.savetxt("Minority_phase_diagram.csv", average_fraction, delimiter=",")

    plt.imshow(average_fraction, origin = 'lower', extent =  [0, 0.3, 0, 0.3])
    plt.savefig("Minority_phase_diagram.png")
    plt.show()
    # np.savetxt("Checkpoint_2/SIRS/phase2.csv", I2, delimiter=",")
    # np.savetxt("Checkpoint_2/SIRS/phase_err.csv", Ierr, delimiter =",")

phase_diagram()