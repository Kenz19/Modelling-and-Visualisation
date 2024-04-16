# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 02:16:07 2024

@author: kenzi
"""

### imports

# modules
import random
import numpy as np

###

def randomise_spins(lattice, p):
    '''
    A function used to randomise the spins within a given lattice. Spins will 
    either be +1 or -1 given a probability threshold of 0.5
    
    Parameters
    ----------
    lattice : Numpy array
        Lattice containing spin up and spin down units.

    Returns
    -------
    lattice : Numpy array
        Lattice containing spin up and spin down units - spins randomised
    
    '''
    
    # shape of lattice
    N1, N2 = lattice.shape
    
    # number used to determine if a spin is up or down based on randomness
    p = 0.5

    # assign each lattice point 0 or 1
    for i in range(N1):
        for j in range(N2):
            r = random.random() # generate random number for each position in the lattice (between 0 and 1)
            if r < p: lattice[i, j] = -1
            if r >= p: lattice[i, j] = 1

    #return lattice


def total_energy(lattice, J):
    '''
    This function measures the total energy of a given lattice
            
    Parameters
    ----------
    lattice : Numpy array
        Lattice containing spin up and spin down units.
        
    J : Float
        Proportionality factor in the hamiltonian

    Returns
    -------
    -energy : float
        total energy of the lattice.
    '''
    
    # size of lattice
    N1, N2 = lattice.shape
    
    # counter for energy
    energy = 0
    
    # for each spin in the lattice, calculate its energy contribution
    for i in range(N1):
        for j in range(N2):
            
            # current spin
            spin = lattice[i,j]
            
            # determine spin of nearest neighbours
            n1 = lattice[i, (j+1)%N2]
            n2 = lattice[(i+1)%N1, j]
            energy += spin*(n1 + n2)
    
    return -energy


def magnetisation(lattice):
    '''
    This function calculates the total magnetisation of a lattice by summing
    each spin unit up in total.

    Parameters
    ----------
    lattice : Numpy array
        Lattice containing spin up and spin down units.

    Returns
    -------
    mag : Float
        total magnetisation of a single lattice.

    '''
    
    # dimensions of lattice
    N1, N2 = lattice.shape
    
    # magnetisation counter
    mag = 0
    
    # for each spin in the lattice
    for i in range(N1):
        for j in range(N2):
            
            # add spin to total magnetisation count
            mag += lattice[i,j]
            
    return mag


def randomise_spins(lattice):
    '''
    A function used to randomise the spins within a given lattice. Spins will 
    either be +1 or -1 given a probability threshold of 0.5
    '''
    
    # shape of lattice
    N1, N2 = lattice.shape
    
    # number used to determine if a spin is up or down based on randomness
    p = 0.5

    # assign each lattice point 0 or 1
    for i in range(N1):
        for j in range(N2):
            r = random.random() # generate random number for each position in the lattice (between 0 and 1)
            if r < p: lattice[i, j] = -1
            if r >= p: lattice[i, j] = 1

    return lattice


def bootstrap_errors(data, N, T, no_k):
    '''
    Calculate errors via the bootstrap method (general function for c and suseptibility calculations)

    Parameters
    ----------
    data : array
        data for error to be calculated for.
    N : integer
        system size.
    T : float
        system temperature.

    Returns
    -------
    error : float
        error on input data.

    '''
    
    # how many measurements in the data
    n = len(data)
    
    values = []
    
    # for k iterations
    for k in range(no_k):   
        
        # pick a random value from the data
        r = (np.random.random(n)*n).astype(int)
        
        # perform resampling
        resample = data[r]
        value = (np.mean(resample**2)-np.mean(resample)**2)/(N*T)
        values.append(value)
    
    # error calculation from the randomly sampled data
    values = np.array(values)
    error = np.sqrt(np.mean(values**2)-np.mean(values)**2)
    
    return error