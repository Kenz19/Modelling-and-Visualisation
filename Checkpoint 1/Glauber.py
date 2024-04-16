'''
MVP CP1

This script contains all of the functions specific to the glauber dynamics 
route of the Ising model.

It contains a function to calculate the energy change and a function 
to run the glauber dynamics simulation with the metropolis algorithm.
'''

### imports

import numpy as np
import random

###

def energy_change_glauber(lattice, i, j):
    '''
    This function calculates the energy change between two lattice states
    using the Glauber flip method. This assumes 4 nearest neighbours

    Parameters
    ----------
    lattice : Numpy array
        Lattice containing spin up and spin down units.
    i : integer
        x position of randomly chosen spin to be flipped on lattice
    j : integer
        y position of randomly chosen spin to be flipped on lattice

    Returns
    -------
    deltaE : float
        energy difference between system nu and system mu.
    '''
    
    # dimensions of lattice
    N1, N2 = lattice.shape
    
    # randomly chosen spin
    spin = lattice[i,j]
    
    # spins of nearest neighbours of spin
    n1 = lattice[i,(j+1)%N2]
    n2 = lattice[i,(j-1)%N2]
    n3 = lattice[(i+1)%N1,j]
    n4 = lattice[(i-1)%N1,j]
    
    #nn_sum = lattice[i,(j+1)%N2] + lattice[i,(j-1)%N2] + lattice[(i+1)%N1,j] + lattice[(i-1)%N1,j]
    deltaE = 2*spin*(n1+n2+n3+n4)
    
    return deltaE


def glauber_dynamics(lattice, T):
    '''
    This function performs a glauber flip and uses the metropolis algorithm
    to determine if the new state is accepted or not. 

    Parameters
    ----------
    lattice : Numpy array
        Lattice containing spin up and spin down units.
    T : float
        temperature of system.

    Returns
    -------
    No return - updates global lattice
    '''
    
    # dimensions of lattice
    N1, N2 = lattice.shape
    
    # choose random spin site on lattice
    i,j = np.random.randint(0, N1), np.random.randint(0, N2) 

    # calculate energy difference between state mu and trial state nu
    deltaE = energy_change_glauber(lattice, i, j)
    
    # perform metropolis algorithm for trail state
    
    # boltzmann weighting
    b = np.exp(-(deltaE)/T)
    
    # determine probability of new state 
    prob = np.minimum(1, b)
    
    # if random number less than our probabilty, accept the new state as current state
    r = np.random.random()
    if(r<prob):
        
        # flip spin
        lattice[i,j] *= -1 
        
    #return lattice