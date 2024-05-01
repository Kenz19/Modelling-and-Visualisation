import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt

# for animations
import matplotlib
matplotlib.use('TKAgg')

class initial_value_problem():

    def __init__(self, N1 = 50, N2 = 50, phi0 = 0.5, noise = 0.1, a = 0.1, b = 0.1, M = 0.1, kappa = 0.1, dx = 1, dt = 1, no_steps = 1e5, animate = False, plot_free_energy = True):
        '''
        Initialise the order parameter grid and other parameters needed to update the order parameter and chemical potential

        N1: integer
            Number of rows present in order parameter grid
        N2: integer
            Number of columns present in the order parameter grid
        phi0: float
            Initial condition value for phi
        noise: float
            Maximum and minimum noise value that the system will be initialised with
        a, b: floats
            Positive constants that ensure that phi = 0 is unstable and phi = sqrt(a/b) is stable
        M: float
            Mobility constant (the equivalent of the diffusion coeffcient for a mixture)
        kappa: float
            Constant related to surface tension of oil-water interface
        dx: float
            Value of spatial discretisation
        dt: float
            Value of time discretisation
        no_steps: integer
            The number of steps to carry out in the simulation (can be input as a float but is converted to an integer)
        animate: boolean
            Either True or False, True: simulation is carried out with displayed animation, False: simulation is carried out without displayed animation and instead measurements of the free energy are carried out.
        plot_free_energy: boolean
            Either True or False, True: Free energy is plotted vs time when the measurements simulation (Animation = False) is carried out, False: Only the free energy data file is output, no plot.
        '''

        # add parameters to self
        self.rows = int(N1) # number of rows in phi grid
        self.cols = int(N2) # number of columns in phi grid
        self.phi0 = phi0 # initial value of phi for initalisation of grid
        self.noise = noise
        self.a = a
        self.b = b # generally convention to set a = b but for generalisabilities sake, will implement b as its own variable
        self.M = M # mobility
        self.k = kappa # constant related to surface tension of oil and water interface

        # discrete steps 
        self.dx = dx # positional
        self.dt = dt # spatial

        # number of steps for simulation
        self.steps = int(no_steps)

        # plotting parameter (True or False)
        self.plot_free_energy = plot_free_energy

        # initialise grid of phi values of size N1 x N2 (Idealling 100 x 100) within some noise
        self.phi = np.random.uniform(self.phi0 - self.noise, self.phi0 + self.noise, (self.rows, self.cols))

        # run simulation with animation if true
        if animate == True:
            self.simulation_animation()

        # now input another condition for the free energy measurements
        else:
            self.simulation_measurements()

    def update_mu(self):
        '''
        Update the chemical potential at each timestep - use central difference scheme

        Use of np.roll implements periodic boundary conditions
        '''
        # chemical potential at timestep n
        self.mu = -(self.a*self.phi) + (self.b*self.phi**3) - (self.k/self.dx**2)*(np.roll(self.phi, 1, axis=0) + np.roll(self.phi, -1, axis = 0)
                                                                                        + np.roll(self.phi, 1, axis = 1) + np.roll(self.phi, -1, axis = 1)
                                                                                        - 4*self.phi)
    
    def laplacian(self):

        laplacian = (np.roll(self.mu, 1, axis = 0) + np.roll(self.mu, -1, axis = 0) + np.roll(self.mu, 1, axis = 1) + np.roll(self.mu, -1, axis = 1) - 4*self.mu)/self.dx**2

        return laplacian
    
    @staticmethod
    def general_laplacian(grid, dx):
        return (np.roll(grid, 1, axis = 0) + np.roll(grid, -1, axis = 0) + np.roll(grid, 1, axis = 1) + np.roll(grid, -1, axis = 1) - 4*grid)/dx**2

    def update_phi(self):
        '''
        Update the order parameter for step.

        Use of np.roll implements periodic boundary conditions
        '''
        # adding the chemical potential contribution of current value of phi
        # self.phi += self.M*(self.dt/self.dx**2)*(np.roll(self.mu, 1, axis = 0) + np.roll(self.mu, -1, axis = 0) + np.roll(self.mu, 1, axis = 1)
        #                                          + np.roll(self.mu, -1, axis = 1) - 4*self.mu)
        
        self.phi +=self.M*self.dt*self.general_laplacian(self.mu, self.dx)

    def simulation_animation(self, visualise = 100):
        '''
        Run the simulation with animations, measure the free energy

        params
        ------
        visualise: integer
            every nth frame to visualise the simulation
        '''

        # set up animation grid
        plt.figure()
        plt.imshow(self.phi, vmax=1, vmin=-1, animated=True, cmap='coolwarm_r') # blue water, red oil
        plt.colorbar()

        # first update mu

        #for i, step in tqdm(enumerate(self.steps+1)):
        for i in tqdm.tqdm(range(self.steps)):
            
            # update chemical potential
            self.update_mu()

            # update phi with new chemical potential
            self.update_phi()

            # visualise every 100th step in the simulation
            if i%visualise == 0:
                plt.cla()   
                plt.title(i) # for indication of what step we are on
                im = plt.imshow(self.phi, vmax=1, vmin=-1, animated=True, interpolation='gaussian', cmap='coolwarm')
                plt.draw()
                plt.pause(0.0001)

    def measure_free_energy(self):
        '''
        Measure the free energy in the simulation as a function of time 

        this function calculates the free energy density of which the free energy is the sum of
        '''

        # central
        f = -(self.a/2)*(self.phi**2) + (self.a/4)*(self.phi**4)+ (self.k/(8*(self.dx**2)))*( 
            (np.roll(self.phi, -1, axis=0) - self.phi)**2 + (np.roll(self.phi, -1, axis=1) - self.phi)**2)
                                                                                        
        return np.sum(f)  


    def simulation_measurements(self, measurements = 100):
        '''
        Runs simulation and outputs a file containing the free energy of the system as a function of time, the trend is plotted if requested

        Measurements: integer
            Every nth frame to measure the free energy (same as visualise in the animation function)
        '''

        # array to store the measured free energy and the time it was measured
        free_energy = np.zeros([int(self.steps/measurements), 2])

        for i in tqdm.tqdm(range(self.steps - 1)):

            # update chemical potential
            self.update_mu()

            # update phi with new chemical potential
            self.update_phi()

            # measure free energy density and write to array if on a desired frame
            if i%measurements == 0:
                free_energy[int(i/measurements), 0] = self.measure_free_energy() # free energy
                free_energy[int(i/measurements), 1] = i # time at which the energy was measured

        # write the free_energy_densities to a file
        np.savetxt(f'Checkpoint_3/free_energies_phi0={self.phi0}.csv', free_energy, delimiter=',')

        # plot if plotting condition is True
        if self.plot_free_energy == True:

            plt.scatter(free_energy[:, 1], free_energy[:, 0], s = 5)
            plt.xlabel('Time')
            plt.ylabel('Free Energy')
            plt.title(f'Free Energy vs Time ($\phi_0$ = {self.phi0})')
            plt.savefig(f'Checkpoint_3/free_energies_phi0={self.phi0}.png')
            plt.show()


def main():

    phi0 = float(input('Please provide a value for phi_o: '))
    N = int(input('Please provide a system size: '))
    type = input('Animation or free energy measurements? (a or m) ')

    if type == 'a':
        test = initial_value_problem(N1 = N, N2 = N, phi0 = phi0, animate = True, plot_free_energy = False)

    if type == 'm':
        test = initial_value_problem(N1 = N, N2 = N, phi0 = phi0, animate = False, plot_free_energy = True)

    else:
        print('Invalid simulation type input, please choose either animation or measurements (a or m)')
        main()

if __name__ == "__main__":
    main()
