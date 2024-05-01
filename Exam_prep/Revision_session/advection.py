import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# for animations
import matplotlib
matplotlib.use('TKAgg')

class Advection():

    def __init__(self, N1 = 50, N2 = 50, phi0 = 0.5, noise = 0.1, D = 1, kappa = 0.01, sigma = 10, v0 = 0.1, dx = 1, dt = 0.001, no_steps = 1e6, animate = False):
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
        D: float
            Diffusion constant (the equivalent of the diffusion coeffcient for a mixture)
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
        self.D = D # diffusion coefficient
        self.k = kappa # constant related to surface tension of oil and water interface
        self.sigma = sigma # constant for determining rho
        self.v0 = v0

        # discrete steps 
        self.dx = dx # positional
        self.dt = dt # spatial

        # number of steps for simulation
        self.steps = int(no_steps)

        # initialise grid of phi values of size N1 x N2 within some noise
        self.phi = np.random.uniform(self.phi0 - self.noise, self.phi0 + self.noise, (self.rows, self.cols))

        # calculate rho
        #self.calculate_rho() # rho remains constant therefore only needs to be initialised once.
        #self.calculate_rho() # rho remains constant therefore only needs to be initialised once.
        self.x, self.y = self.dx*(np.mgrid[-self.rows//2:self.cols//2, -self.rows//2:self.cols//2] + self.rows % 2)
        self.rho = np.exp(-(self.x*self.x + self.y*self.y) / self.sigma**2)

        # initialise vx
        self.compute_vx()

        # run simulation with animation if true
        if animate == True:
            self.simulation_animation()



            #self.calculate_rho()
            #print('Nothing Happened')

    # def update_mu(self):
    #     '''
    #     Update the chemical potential at each timestep - use central difference scheme

    #     Use of np.roll implements periodic boundary conditions
    #     '''
    #     # chemical potential at timestep n
    #     self.mu = -(self.a*self.phi) + (self.b*self.phi**3) - (self.k/self.dx**2)*(np.roll(self.phi, 1, axis=0) + np.roll(self.phi, -1, axis = 0)
    #                                                                                     + np.roll(self.phi, 1, axis = 1) + np.roll(self.phi, -1, axis = 1)
    #                                                                                     - 4*self.phi)
        
    def calculate_rho(self):
        '''
        Generates a rho matrix which remains constant at each n.

        takes self.phi and generates a matrix with the distances from the centre of the matrix for each cell
        '''

        # central i, j coordinates
        i_centre = int(self.rows/2)
        j_centre = int(self.cols/2)

        # grids of x and y (i and j) coordinates of each 
        x_arr, y_arr = np.mgrid[0:self.rows, 0:self.cols] 

        # distance from the centre of phi array for each cell
        self.r = np.sqrt((x_arr - i_centre)**2 + (y_arr - j_centre)**2)

        self.rho = np.exp(-(self.r**2)/self.sigma**2) # initialise rho matrix into the class

        # plt.imshow(self.rho)
        # plt.show()

        #print(self.rho)

    def compute_vx(self):
        '''
        Compute vx
        '''

        # x and y coordinates (be careful as technically row number, col number)
        #y_arr, x_arr = np.mgrid[0:self.rows, 0:self.cols] 

        self.vx = -self.v0*np.sin(2*np.pi*self.y/self.rows) # assuming self.rows = self.cols

        
    def update_phi(self):
        '''
        Phi with an additional velocity term added on
        '''
        self.phi += self.D*(self.dt/self.dx**2)*(np.roll(np.copy(self.phi), 1, axis = 0) + np.roll(np.copy(self.phi), -1, axis = 0) + np.roll(np.copy(self.phi), 1, axis = 1)
                                                 + np.roll(np.copy(self.phi), -1, axis = 1) - 4*np.copy(self.phi)) + self.dt*(self.rho - self.k*np.copy(self.phi) - self.vx*np.gradient(self.phi)[0])
        
        # plt.imshow(self.phi)
        # plt.show

    def simulation_animation(self, visualise = 100):
        '''
        Run the simulation with animations
        
        params
        ------
        visualise: integer
            every nth frame to visualise the simulation
        '''

        # set up animation grid
        plt.figure()
        im = plt.imshow(self.phi, animated=True, cmap='coolwarm_r', interpolation = 'gaussian') # blue water, red oil
        cbar = plt.colorbar(im)

        #plt.imshow(self.phi, animated=True, cmap='coolwarm_r')
        #plt.colorbar(im)

        # first update mu

        #for i, step in tqdm(enumerate(self.steps+1)):
        for i in tqdm.tqdm(range(self.steps)):

            # update phi with new chemical potential
            self.update_phi()
            print(self.phi)

            # visualise every 100th step in the simulation
            if i%visualise == 0:
                plt.cla()   
                plt.title(i) # for indication of what step we are on
                im = plt.imshow(self.phi, animated=True, cmap='coolwarm_r', interpolation = 'gaussian')#, interpolation='gaussian')
                cbar.update_normal(im) # update the colour bar to match the current displayed image in the animation
                plt.draw()
                plt.pause(0.0001)


    def average_phi(self, measurements = 1000):
        '''
        Calculates the average value of phi over time, a plot is then produced of this trend
        '''

        # lists to store average phi and time in the 
        average_phi = []
        time = []

        # get average phi of initialisation
        average_phi.append(np.sum(self.phi)/(self.rows*self.cols))
        time.append(0)

        for i in tqdm.tqdm(range(self.steps)):
            
            # update phi value
            self.update_phi()

            if i%measurements == 0:

                # get average value of phi 
                average_phi.append(np.sum(self.phi)/(self.rows*self.cols))
                time.append(i+1)


        # combine data into an array and save
        combined_array = np.column_stack((np.array(average_phi), np.array(time)))
        np.savetxt(f'Average_phi(k={self.k},sigma={self.sigma}).csv', combined_array, delimiter = ';')

        # plot and save plot
        plt.scatter(time, average_phi)
        plt.xlabel('Time')
        plt.ylabel('$\langle \phi \\rangle$')
        plt.title('Average $\phi$ with time')
        plt.savefig(f'Average_phi(k={self.k},sigma={self.sigma}).png')
        plt.show()



    # def plot_phi_vs_r(self):

    #     # flatten r array to get a 1d r array
    #     r_flat = self.r.flatten()

    #     # run simulation and get the final phi (as this is in steady state)
    #     for i in tqdm.tqdm(range(self.steps)):
            
    #         # update phi value
    #         self.update_phi()

    #     phi_flat = self.phi.flatten()

    #     plt.scatter(r_flat, phi_flat) 
    #     plt.show()









    # def measure_free_energy(self):
    #     '''
    #     Measure the free energy in the simulation as a function of time 

    #     this function calculates the free energy density of which the free energy is the sum of
    #     '''

    #     # central
    #     f = -(self.a/2)*(self.phi**2) + (self.a/4)*(self.phi**4)+ (self.k/(8*(self.dx**2)))*( 
    #         (np.roll(self.phi, -1, axis=0) - self.phi)**2 + (np.roll(self.phi, -1, axis=1) - self.phi)**2)
                                                                                        
    #     return np.sum(f)  


    # def simulation_measurements(self, measurements = 100):
    #     '''
    #     Runs simulation and outputs a file containing the free energy of the system as a function of time, the trend is plotted if requested

    #     Measurements: integer
    #         Every nth frame to measure the free energy (same as visualise in the animation function)
    #     '''

    #     # array to store the measured free energy and the time it was measured
    #     free_energy = np.zeros([int(self.steps/measurements), 2])

    #     for i in tqdm.tqdm(range(self.steps - 1)):

    #         # update chemical potential
    #         self.update_mu()

    #         # update phi with new chemical potential
    #         self.update_phi()

    #         # measure free energy density and write to array if on a desired frame
    #         if i%measurements == 0:
    #             free_energy[int(i/measurements), 0] = self.measure_free_energy() # free energy
    #             free_energy[int(i/measurements), 1] = i # time at which the energy was measured

    #     # write the free_energy_densities to a file
    #     np.savetxt(f'Checkpoint_3/free_energies_phi0={self.phi0}.csv', free_energy, delimiter=',')

    #     # plot if plotting condition is True
    #     if self.plot_free_energy == True:

    #         plt.scatter(free_energy[:, 1], free_energy[:, 0], s = 5)
    #         plt.xlabel('Time')
    #         plt.ylabel('Free Energy')
    #         plt.title(f'Free Energy vs Time ($\phi_0$ = {self.phi0})')
    #         plt.savefig(f'Checkpoint_3/free_energies_phi0={self.phi0}.png')
    #         plt.show()


def main():

    kappa = float(input('Please provide a value for kappa: '))
    sigma = int(input('Please provide a value for sigma: '))

    # if type == 'a':
    #     test = initial_value_problem(N1 = N, N2 = N, phi0 = phi0, animate = True, plot_free_energy = False)

    # if type == 'm':
    #     test = initial_value_problem(N1 = N, N2 = N, phi0 = phi0, animate = False, plot_free_energy = True)

    # else:
    #     print('Invalid simulation type input, please choose either animation or measurements (a or m)')
    #     main()
    test = Advection(kappa = kappa, sigma = sigma, v0 = 0.5, animate = True)
    #test.compute_v0_field()

if __name__ == "__main__":
    main()
