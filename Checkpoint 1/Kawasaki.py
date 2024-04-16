"""
MVP CP1

This script contains all of the functions specific to the kawasaki dynamics 
route of the Ising model.

It contains a function to run the kawasaki dynamics simulation with the 
metropolis algorithm.
"""

### imports

# modules
import numpy as np

# external files
import Glauber as Gl

###

def kawasaki_dynamics(lattice, T):
    '''
    This function performs a kawasaki exchange and uses the metropolis algorithm
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
    
    # choose a random spin within the lattice - flip this to give the trial state
    x, y = np.random.randint(0, N1), np.random.randint(0, N2) # 1st spin
    w, z = np.random.randint(0, N1), np.random.randint(0, N2) # 2nd spin
    
    # if sites hold the same spin value pick another set of sites, continue
    # continue until sites of opposite spin have been chosen
    while lattice[x,y] == lattice[w,z]: 
        
        # choose sites again 
        x, y = np.random.randint(0, N1), np.random.randint(0, N2) # 1st spin
        w, z = np.random.randint(0, N1), np.random.randint(0, N2) # 2nd spin
    
    # calculate the energy change of both sites flipping as glauber flips
    deltaE1 = Gl.energy_change_glauber(lattice, x, y)  
    deltaE2 = Gl.energy_change_glauber(lattice, w, z)   
    
    # total energy change
    deltaE = deltaE1 + deltaE2
    
    # check if correction is needed for selected spins being nearest neighbours.
    # check horizontal and vertical directions. If boolean values turn true,
    # picked spins were nearest neighbours.
    hnn = (x==w and np.abs(y-z)==1)
    vnn = (np.abs(x-w)==1 and w==z) 
    
    # apply corrections if random spins were found as nearest neighbours
    if hnn or vnn:
        deltaE += 4    
     
    # apply metropolis algorithm to the new state

    # boltzmann weight 
    b = np.exp(-(deltaE)/T) 
    
    # probability 
    p = np.minimum(1, b)   
    
    # random number for acceptance clause
    r = np.random.random()
    
    # if random number less than probabilty, swap the two spin sites
    if r < p:
        lattice[x,y] *= -1  
        lattice[w,z] *= -1 
        
    #return lattice 

