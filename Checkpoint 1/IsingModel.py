"""

"""

### imports

# modules
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

import matplotlib as mpl
mpl.use('TKAgg')

# external files
import Functions as func # general functions
import Glauber as Gl # glauber functions
import Kawasaki as Ka

###

# specify stylesheet I use
#plt.style.use('shendrukGroupStyle') # i have this line hashed out as it wont work on others computers but I used it to generate the nice plots

def quantitative_measurements(lattice, dynamics, J, Tmax, Tmin, step, no_sweeps):   
    '''
    This function measures the quantitative values needed for tasks 4 and 5.
    
    This includes the magnetisation, susceptibility, energy & scaled specific
    heat
    '''
    
    # temperature range of which to iterate over
    temperatures = np.arange(Tmin, Tmax+step, step)
    
    # length of lattice (assuming shape N x N)
    N1, N2 = lattice.shape
    no_spins = N1*N2
    
    # arrays to store the observables and the corresponding errors
    avgE = []
    errE = []
    E2 = []
    errCv = []
    avgM = []
    M2 = []
    
    for T in temperatures:
    
        print('T = ' + str(T) + ' out of ' + str(Tmax))
    
        # energies and magntitude per sweep
        energies = []
        mags = []
        
        # for each sweep
        for sweep in range(no_sweeps):
            
            # each sweep has the number of moves equivalent to the number of spins
            for i in range(no_spins):
                
                # user input of dynamics type determines which algorithm to run
                if dynamics=='g':
                    # perform glauber dynamics
                    lattice = Gl.glauber_dynamics(lattice, T) 
                    
                if dynamics=='k':
                    # perform kawasaki dynamics
                    lattice = Ka.kawasaki_dynamics(lattice, T)

            # check if system has reached equilibrium, if so, take a measurement if states are uncorrelated (every 10th state)
            if sweep>100 and (sweep%10 == 0):
                
                # record energies and magnetisation
                energies.append(func.total_energy(lattice, J))
                mags.append(np.abs(func.magnetisation(lattice)))
        
        # task 4 measurements
        avgM.append(np.mean(mags)) # average magnetisation for this temperature
        M2.append(np.mean(np.array(mags)**2)) 
        
        # task 5 measurements
        avgE.append(np.mean(energies))
        errE.append(np.std(energies)/np.sqrt(len(energies)))
        E2.append(np.mean(np.array(energies)**2)) 
        errCv.append(func.bootstrap_errors(np.array(energies), no_spins, T**2, 1000))
     
    # convert magnetisation and energy to arrays
    avgE = np.array(avgE)
    avgM = np.array(avgM)
    
    generate_plots(temperatures, avgE, errE, avgM, E2, errCv, M2, no_spins)

    
# A function for producing the desired plots and outputting the data on a datafile.
def generate_plots(T, E, E_err, M, Esqrd, c_err, Msqrd, N):
    
    # calculate heat capacity
    c = (Esqrd - (E**2))/(N*T**2)
    
    # calcualte susceptibility
    chi = (Msqrd - (M**2))/(N*T)  
    
    f = open('datafile.txt','w')   
    titles = 'T, <E>, E_error, <|M|>, c, c_error, Chi'
    
    # write all data to file
    f.write('%s\n\n'%(titles))  

    for i in range(len(T)):

        # format and write
        f.write('%lf %lf %lf %lf %lf %lf %lf\n'%(T[i], E[i], E_err[i], M[i], c[i], c_err[i], chi[i]))

    f.close() # close to avoid overwrites
    
    # plot energy
    plt.errorbar(T, E, yerr=E_err, fmt='ro', )
    plt.xlabel("Temperature, $T$")
    plt.ylabel("Average energy, $\langle E \\rangle$")
    plt.savefig('Energy.png')
    plt.show()
    
    # plot specific heat
    plt.errorbar(T, c , yerr=c_err, fmt='ro')
    plt.xlabel("Temperature, $T$")
    plt.ylabel("Scaled Heat Capacity, $c$")
    plt.savefig('ScaledHeatCapacity.png')
    plt.show()
    
    # plot magnetisation 
    plt.scatter(T, M)
    plt.xlabel("Temperature, $T$")
    plt.ylabel("$\langle |M| \\rangle$")
    plt.savefig('Magnetisation.png')
    plt.show()
    
    # plot susceptibility
    plt.scatter(T, chi)
    plt.xlabel("Temperature, T")
    plt.ylabel("Susceptibility, $\chi$")
    plt.savefig('Susceptibility.png')
    plt.show()
    

def animation(lattice, dynamics, T):
    """
    Animate a simulation of the Ising model.

    Parameters
    ----------
    lattice : numpy array
        initial state

    dynamics : string
        either 'g' for glauber dynamics or 'k' for kawasaki dynamics

    T : float
        temperature at which to run simulation
    """

    # reset style sheet
    mpl.rcParams.update(mpl.rcParamsDefault)

    # lattice dimensions
    N1, N2 = lattice.shape

    # initialise animations figure
    fig = plt.figure()
    #im = plt.imshow(lattice, animated=True)

    # for each sweep
    for sweep in range(10000):
        
        # each sweep has the number of moves equivalent to the number of spins
        for flip in range(N1*N2):
            
            # user input of dynamics type determines which algorithm to run
            if dynamics == 'g':
                # perform glauber dynamics
                #lattice = Gl.glauber_dynamics(lattice, T)
                Gl.glauber_dynamics(lattice, T) 

            if dynamics == 'k':
                # perform kawasaki dynamics
                #lattice = Ka.kawasaki_dynamics(lattice, T)
                Ka.kawasaki_dynamics(lattice, T)

        # visualise every 10th state even outside of equilibrium
        if sweep % 10 == 0:
            
            plt.cla()
            im = plt.imshow(lattice, animated=True)
            plt.draw()
            plt.pause(0.0001)
            
def main(J): 
    '''
    This is the main function of the script containing the user inputs. First
    the option of measurments or animation is given and then further prompts from 
    there, both paths allow a choice of dynamics.
    '''
    
    # prompt for measurements or animation to be carried out
    choice = input("Animations or Quantitative measurements? (a or m): ")

    # choose to measure
    if choice == 'm':
        
        # prompt for variables
        N = int(input("System size N (integer) : ")) # system size
        dynamics = input("Glauber or kawasaki Dynamics (g or k): ")
        
        # initialise global lattice in ground state for glauber
        lattice = np.ones([N, N])
 
        # if kawaski was chosen move to a more disorderd state for the ground state
        if dynamics == 'k':
            
            # scramble spins out of low entropy state
            func.randomise_spins(lattice, 0.5)

        # run measurements and plot quantities
        quantitative_measurements(lattice, dynamics, J, 3, 1, 0.1, 10000)   # run the measurements code
    
    # choose to animate
    elif choice == 'a':
        
        # prompt for variables
        N = int(input("Type in the system size N: ")) # system size
        T = float(input("Type in the temperature T: ")) # temperature
        dynamics = input("Type in the desired dynamics to be used\n('g' for Glauber or 'k' for Kawasaki): ") # dynamics
        
        # initialise lattice
        lattice = np.ones((N,N))
        
        lattice = func.randomise_spins(lattice)   # randomise the spins of the initial lattice

        animation(lattice, dynamics, T)   # run the animation code
        
    else:
        print("That isnt an option, please input 'a' for animation or 'm' for measurements")
        main()


# run the main function
if __name__ == "__main__":

    main(J = 1)
