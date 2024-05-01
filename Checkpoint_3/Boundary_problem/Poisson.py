import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

class Poisson():

    def __init__(self, N1 = 50, N2 = 50, N3 = 50, field = 'E', algorithm = 'j', tol = 1e-3):
        '''
        N1: integer
            Number of rows present in order parameter grid
        N2: integer
            Number of columns present in the order parameter grid
        N3: integer
            Depth of parameter grid
        field: string
            Two choices: E or M. E: study electric fields, a point charge is initalised at the centre of the charge density grid. M: studies
            magnetic fields, a line of charge is initalised down the centre of the x-y plane along the z axis, acting as a wire.
        algorithm: string
            Three choices of updater algorithm: 1. Jacobian algorithm (j), 2: Gauss-Seidel algorithm (g), 3: Successive over relaxation (s) this runs an optimisation test for omega instead of a single simulation.
        tol: float
            Tolerance for convergence, once a difference measure is smaller than this tolerance the simulation will be considered converged.
        '''

        # system dimensions
        self.rows = int(N1)
        self.cols = int(N2)
        self.depth = int(N3)

        # initialise phi grid
        self.phi = np.zeros((self.rows, self.cols, self.depth))

        # initialise charge density grid
        self.rho = np.zeros((self.rows, self.cols, self.depth))

        self.field = field

        # # initialise a point charge at the centre of rho if looking at an electric field study
        # if self.field == 'E':

        #     self.phi = np.zeros((self.rows, self.cols, self.depth))

        #     # initialise charge density grid
        #     self.rho = np.zeros((self.rows, self.cols, self.depth))

        #     self.rho[int(self.rows/2), int(self.cols/2), int(self.depth/2)] = 1

        # # initialise central row along the z axis, like a wire for magnetism
        # elif self.field == 'M':

        #     self.phi = np.zeros((self.rows, self.cols))

        #     # initialise charge density grid
        #     self.rho = np.zeros((self.rows, self.cols))

        #     self.rho[int(self.rows/2), int(self.cols/2)] = 1


        # initialise a point charge at the centre of rho if looking at an electric field study
        if self.field == 'E':
            self.rho[int(self.rows/2), int(self.cols/2), int(self.depth/2)] = 1

        # initialise central row along the z axis, like a wire for magnetism
        if self.field == 'M':
            self.rho[int(self.rows/2), int(self.cols/2), :] = 1



        # error tolerance
        self.tol = float(tol)

        # type of algorithm
        self.algorithm_type = algorithm

        # run the simulation
        self.run_simulation()


    def enforce_boundaries(self, grid):
        '''
        Fix boundaries at the edge of a box, represented by an input array to zero. This is typically done when
        using the Dirichlet condition.

        params
        ------
        grid: array (3D)
            Grid containing phi values at a point in time

        ''' 
        grid[0,:,:] = 0 # edge beginning along x axis
        grid[:,0,:] = 0 # edge beginning along y axis
        grid[:,:,0] = 0 # edge beginning along z axis  

        grid[self.rows-1,:,:] = 0 # edge ending along x axis
        grid[:,self.cols-1,:] = 0 # edge ending along y axis
        grid[:,:,self.depth-1] = 0 # edge ending along z axis

        return grid
    
    def checkerboard_mask(self):  
        '''
        Create a 3d boolean mask of a checkerboard style, (True, False, True, False etc)
        '''

        # Create a 3D array filled with False values
        mask_3d = np.full((self.depth, self.rows, self.cols), False, dtype=bool)

        # Create the checkerboard mask for a single 2D slice
        mask = np.full((self.rows, self.cols), False, dtype=bool)
        mask[::2, ::2] = True
        mask[1::2, 1::2] = True

        # Assign the checkerboard mask to each 2D slice in the 3D array
        for i in range(self.depth):
            mask_3d[i] = mask

        return mask_3d, np.invert(mask_3d) 

    # algorithms

    def update_jacobi(self):
        '''
        Discretised update rule using the jacobi algorithm in 3D for the poisson equation.

        A difference measure between phi and new_phi is measured as a mechanism for stopping an iterative algorithm
        '''         

        # update new phi
        self.phi_new = (1/6)*(self.enforce_boundaries(np.roll(self.phi,-1,axis=0)) + self.enforce_boundaries(np.roll(self.phi,+1,axis=0)) + self.enforce_boundaries(np.roll(self.phi,-1,axis=1)) + 
                              self.enforce_boundaries(np.roll(self.phi,+1,axis=1)) + self.enforce_boundaries(np.roll(self.phi,-1,axis=2)) + self.enforce_boundaries(np.roll(self.phi,+1,axis=2)) + self.rho)
        
        # difference measure between new phi and the old phi
        self.phi_difference = np.sum(np.abs(self.phi_new - self.phi))

        # replace the old phi with the new one
        self.phi = self.phi_new


    # def update_GS_alternative(self): #(slow)
    #     '''
    #     Discretised update rule using the Gauss-Seidel in 3D for the poisson equation.

    #     This uses the most recent value for each timestep, as opposed to updating the 
    #     whole lattice from the old values.

    #     - !!! try and update this to a checkerboard update mechanism for efficiency before exam
    #     '''

    #     # copy the current phi array for difference comparison update
    #     copy = np.copy(self.phi)

    #     # loop over each charge in phi and update it using Gauss-Seidel algorithm
    #     for i in range(1, self.rows-1): # x axis
    #         for j in range(1, self.cols-1): # y axis
    #             for k in range(1, self.depth-1): # z axis
    #                 self.phi[i,j,k] = (1/6)*(self.phi[i-1,j,k] + self.phi[i+1,j,k] + self.phi[i,j-1,k] 
    #                                          + self.phi[i,j+1,k] + self.phi[i,j,k-1] + self.phi[i,j,k+1] + self.rho[i,j,k])
                    
    #     # difference measure between new phi and old phi
    #     self.phi_difference = np.sum(np.abs(copy - self.phi))

    def update_GS(self): # attempt at checkerboard - better than looped version
        '''
        Discretised update rule using the Gauss-Seidel in 3D for the poisson equation.

        This uses the most recent value for each timestep, as opposed to updating the 
        whole lattice from the old values.

        Use a checkerboard algorithm for updating.
        '''

        copy = np.copy(self.phi)

        black, white = self.checkerboard_mask()

        # first update all the 'white' squares on the board, at each z
        self.phi = (1/6)*(self.enforce_boundaries(np.roll(self.phi, 1, axis = 0)) + self.enforce_boundaries(np.roll(self.phi, -1, axis = 0)) + self.enforce_boundaries(np.roll(self.phi, 1, axis = 1)) + 
                          self.enforce_boundaries(np.roll(self.phi, -1, axis = 1)) + self.enforce_boundaries(np.roll(self.phi, 1, axis = 2)) + self.enforce_boundaries(np.roll(self.phi, -1, axis = 2)) + self.rho)

        # replace updated black squares with previous values
        self.phi[black] = copy[black]

        # save copy of phi where only the white squares have been updates
        copy_white = np.copy(self.phi)

        # now update all of the 'black' squares on the board for each value of z
        self.phi = (1/6)*(self.enforce_boundaries(np.roll(self.phi, 1, axis = 0)) + self.enforce_boundaries(np.roll(self.phi, -1, axis = 0)) + self.enforce_boundaries(np.roll(self.phi, 1, axis = 1)) + 
                          self.enforce_boundaries(np.roll(self.phi, -1, axis = 1)) + self.enforce_boundaries(np.roll(self.phi, 1, axis = 2)) + self.enforce_boundaries(np.roll(self.phi, -1, axis = 2)) + self.rho)

        # replace updated white squares with previous values (on the already updated white grid)
        self.phi[white] = copy_white[white]

        # calculate difference between updated array from the checkerboard method with the original array
        self.phi_difference = np.sum(np.abs(copy - self.phi))

    # def update_SOR(self, omega): - not working (attempt at checkerboard method)
    #     '''
    #     Discretised update rule using the Gauss-Seidel in 3D for the poisson equation.

    #     This uses the most recent value for each timestep, as opposed to updating the 
    #     whole lattice from the old values.

    #     Use a checkerboard algorithm for updating.

    #     params
    #     ------
    #     omega: float
    #         SOR value, between 0 and 2. Otherwise system will not converge
    #     '''

    #     copy_phi = np.copy(self.phi)

    #     black, white = self.checkerboard_mask()

    #     # first update all the 'white' squares on the board, at each z
    #     self.phi = (1/6)*(np.roll(self.phi, 1, axis = 0) + np.roll(self.phi, -1, axis = 0) + np.roll(self.phi, 1, axis = 1) + 
    #                       np.roll(self.phi, -1, axis = 1) + np.roll(self.phi, 1, axis = 2) + np.roll(self.phi, -1, axis = 2) + self.rho)

    #     self.phi *= omega
    #     self.phi += copy_phi*(1 - omega)
    #     #self.phi = self.phi + omega * np.sum(self.phi - copy_phi)

    #     # replace updated black squares with previous values
    #     self.phi[white] = copy_phi[white]

    #     # apply boundary conditions
    #     self.phi = self.enforce_boundaries(self.phi)

    #     # save copy of phi where only the white squares have been updates
    #     copy_white = np.copy(self.phi)

    #     # now update all of the 'black' squares on the board for each value of z
    #     self.phi = (1/6)*(np.roll(self.phi, 1, axis = 0) + np.roll(self.phi, -1, axis = 0) + np.roll(self.phi, 1, axis = 1) + 
    #                       np.roll(self.phi, -1, axis = 1) + np.roll(self.phi, 1, axis = 2) + np.roll(self.phi, -1, axis = 2) + self.rho)

    #     self.phi *= omega
    #     self.phi += copy_phi*(1 - omega)
    #     #self.phi = self.phi + omega * np.sum(self.phi - copy_phi)

    #     # replace updated white squares with previous values (on the already updated white grid)
    #     self.phi[black] = copy_white[black]

    #     # boundary conditions
    #     self.phi = self.enforce_boundaries(self.phi)

    #     # calculate difference between updated array from the checkerboard method with the original array
    #     #self.phi_difference = np.sum(np.abs(copy_phi - self.phi)) 

    # def update_GS(self): # takes too long to run - something wrong
    #     '''
    #     Discretised update rule using the Gauss-Seidel in 3D for the poisson equation.

    #     This uses the most recent value for each timestep, as opposed to updating the 
    #     whole lattice from the old values.

    #     (idealling use checkerboard for speed)
    #     '''
    #     # loop over every row
    #     for i in range(1, self.rows-1):
    #         # loop over every column
    #         for j in range(1, self.cols-1):
    #             # loop down the z axis
    #             for k in range(1, self.depth-1):

    #                 # update phi based on the GS algorithm
    #                 self.phi[i,j,k] = (1/6)*(self.phi[i-1, j, k] + self.phi[i+1, j, k] + self.phi[i, j-1, k] + self.phi[i, j+1, k] + self.phi[i, j, k-1] + self.phi[i, j, k+1] + self.rho[i, j, k])
    
    def update_SOR(self):
        '''
        Discretised update rule using the Gauss-Seidel in 2D for the poisson equation - using successive over-relaxation.- FOR ELECTRIC FIELD updator

        This uses the most recent value for each timestep, as opposed to updating the 
        whole lattice from the old values.

        This is done in 2D for speed as only used for optimising omega. - doing it in 3D would add two additional iterative terms and k to all the indicies (inside a third for loop)
        '''
        
        # loop over each row in the lattice
        for i in range(1, self.rows-1):

            # loop over each column
            for j in range(1, self.cols-1):
                
                # update potential using SOR algorithm
                self.phi[i,j] = (1-self.omega)*(self.phi[i,j]) + (self.omega)*(1/4)*(self.phi[i-1,j] + self.phi[i+1,j] + self.phi[i,j-1] + self.phi[i,j+1] + self.rho[i,j,int(self.depth/2)])  # fixed k, boundaries?
                

    # compute fields

    def compute_E_field(self):
        '''
        Update the electric field of the lattice

        \vec{E} = -\vec{∇}\phi
        '''

        # take negative gradient of phi
        self.Ex, self.Ey, self.Ez = np.gradient(-self.phi, edge_order=2) # play with edge order and see what works best


    def compute_B_field(self):
        '''
        Update the magnetic field of the lattice, from the curl of the potential, A (phi in this script)

        \vec{B} = \vec{∇} x \vec{A}
        '''

        # curl components of the magnetic field
        self.Ex = (1/2)*(self.enforce_boundaries(np.roll(self.phi, -1, axis = 1)) - self.enforce_boundaries(np.roll(self.phi, +1, axis = 1)))
        self.Ey = (-1/2)*(self.enforce_boundaries(np.roll(self.phi, -1, axis = 0)) - self.enforce_boundaries(np.roll(self.phi, +1, axis = 0)))
        #self.Ex = (1/2)*(np.roll(self.phi, -1, axis = 1) - np.roll(self.phi, +1, axis = 1))
        #self.Ey = (-1/2)*(np.roll(self.phi, -1, axis = 0) - np.roll(self.phi, +1, axis = 0))


    # plotting functions

    def plot_phi_contour(self):
        '''
        Plot a contour plot of a slice down the middle of the z-axis of the potential (showing the x-y plane).

        A file containing the numerical values for this slice is also output.
        '''

        slice = self.phi[:, :, int(self.depth/2)]
        
        # take a slice down the centre of the phi array and plot
        plt.imshow(slice)

        # decorators
        plt.xlabel('x')
        plt.ylabel('y')

        if self.field == 'E':
            plt.title(f'Contour plot of the electric potential\nfor a {self.rows}x{self.cols} lattice')

            # save file and plot
            np.savetxt(f'Electric_potential_final_contour_({self.rows}x{self.cols},tol={self.tol}).txt', slice)
            plt.savefig(f'Electric_potential_final_contour_({self.rows}x{self.cols},tol={self.tol}).png')

        if self.field == 'M':
            plt.title(f'Contour plot of the magnetic potential\nfor a {self.rows}x{self.cols} lattice')

            # save file and plot
            np.savetxt(f'Magnetic_potential_final_contour_({self.rows}x{self.cols},tol={self.tol}).txt', slice)
            plt.savefig(f'Magnetic_potential_final_contour_({self.rows}x{self.cols},tol={self.tol}).png')

        plt.show()  
     

    def plot_E_field(self, normalise = True):
        '''
        Plot a vector field representing the electric field, output a data file with potential and electric field for 
        each point in the cube
        '''

        # create grid of x, y and z positions
        x, y, z = np.mgrid[-self.rows//2:self.rows//2, -self.cols//2:self.cols//2, -self.cols//2:self.cols//2] + self.rows%2

        # set normalisation factor for the electric field
        if normalise == True:
            E = np.sqrt(self.Ex**2 + self.Ey**2 + self.Ez**2) # magnitude of electric field if normalisation is required

        else: # 1 if normalisation is not required
            E = 1

        # normalise electric components with normalisation factor
        Ex, Ey, Ez = self.Ex/E, self.Ey/E, self.Ez/E

        # set middle vector magnitude to 0 down the z axis, as point charge
        Ey[int(self.rows/2), int(self.cols/2), :] = 0
        Ex[int(self.rows/2), int(self.cols/2), :] = 0

        # plot vector field of a slice down the z axis
        plt.quiver(x[:, :, self.depth//2], y[:, :, self.depth//2], Ex[:, :, self.depth//2], Ey[:, :, self.depth//2])

        # decorators
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Electric field as a result of the electric potential (slice down the middle of z)')

        # compile and save all electric field data for a single slice through the middle of z, format so decimal places arent crazy large
        compile_data = [x, y, z, self.phi, Ex, Ey, Ez]
        arr = np.stack([arr.flatten() for arr in compile_data]).T # flatten 2d and 3d into 1d arrays that can be output into a text file
        column_titles = "x | y | z | phi | Ex | Ey | Ez"
        np.savetxt(f'Electric_field_poisson_output({self.rows}x{self.cols},tol={self.tol}).txt', arr, header = column_titles, fmt=["%4d", "%4d", "%4d", "%6e", "%6e", "%6e", "%6e"])

        # show electric field figure
        plt.savefig(f'Electric_field_poisson_output({self.rows}x{self.cols},tol={self.tol}).png')
        plt.show()


    def plot_B_field(self, normalise = True):
        '''
        Plot a vector field representing the magnetic field, output a data file with potential and magnetic field for 
        each point in the cube
        '''

        # create grid of x, y and z positions
        x, y, z = np.mgrid[-self.rows//2:self.rows//2, -self.cols//2:self.cols//2, -self.cols//2:self.cols//2] + self.rows%2

        # set normalisation factor for the electric field
        if normalise == True:
            A = np.sqrt(self.Ex**2 + self.Ey**2) # magnitude of electric field if normalisation is required

        else: # 1 if normalisation is not required
            A = 1

        # normalise electric components with normalisation factor
        Ax, Ay = self.Ex/A, self.Ey/A

        # set middle vector magnitude to 0 down the z axis, as point charge
        Ay[int(self.rows/2), int(self.cols/2), :] = 0
        Ax[int(self.rows/2), int(self.cols/2), :] = 0

        # plot vector field of a slice down the z axis
        plt.quiver(x[:, :, self.depth//2], y[:, :, self.depth//2], Ax[:, :, self.depth//2], Ay[:, :, self.depth//2])

        # decorators
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Magnetic field as a result of the magnetic potential (slice down the middle of z)')

        # compile and save all magnetic field data for a single slice through the middle of z, format so decimal places arent crazy large
        compile_data = [x[:, :, self.depth//2], y[:, :, self.depth//2], self.phi[:, :, self.depth//2], Ax[:, :, self.depth//2], Ay[:, :, self.depth//2]]
        arr = np.stack([arr.flatten() for arr in compile_data]).T # flatten 2d and 3d into 1d arrays that can be output into a text file
        column_titles = "x | y | phi | Ax | Ay "
        np.savetxt(f'Magnetic_field_poisson_output({self.rows}x{self.cols},tol={self.tol}).txt', arr, header = column_titles, fmt=["%4d", "%4d", "%6e", "%6e", "%6e"])

        # show electric field figure
        plt.savefig(f'Magnetic_field_poisson_output({self.rows}x{self.cols},tol={self.tol}).png')
        plt.show()

    # # further analysis - first attempt - not working
    # def omega_analysis(self):
    #     '''
    #     Runs numerous
    #     '''

    #     # omegas to iterate over
    #     omega = np.arange(1, 2 + 0.1, 0.1)

    #     no_iterations = np.zeros_like(omega) # to convergence

    #     copy = np.copy(self.phi)
    #     print(copy)

    #     # run simulation for each omega
    #     for i in tqdm.tqdm(range(len(omega))):

    #         count = 0 # count to track number of iterations that have occured in the simulation

    #         while self.phi_difference > self.tol:
    #             count += 1
    #             print(count)
    #             #print(omega)
    #             #self.update_SOR(omega = omega[i]) # run successive over relaxation algorithm
    #             self.omega = omega[i]
    #             self.SOR()
    #             #self.phi = self.sor(self.phi, self.rows, self.rho, 1, omega[i])
    #             #self.phi_difference = np.sum(np.abs(copy - self.phi))

    #             if count > 10000:   # prevent infinite while loop                    
    #                 break

    #         # record number of iterations
    #         no_iterations[i] = count
    #         print(count, self.phi_difference)
    #         self.phi = copy
    #         self.phi_difference = 1000 # make difference big again so convergence can happen again

    #         #print(self.phi)

    #     # compile omega and outputs into a file for saving
    #     data = np.vstack((omega, no_iterations)).T
    #     np.savetxt('omega_convergence_iterations.txt', data)

    #     plt.scatter(omega, no_iterations)
    #     plt.xlabel('$\omega$')
    #     plt.ylabel('Number of iterations')

    #     plt.show()

    def omega_analysis(self):
        '''
        Run the successive over relaxation algorithm for a range of omega. Plot 
        the number of iterations it takes for convergence to occur as a function of omega.
        '''                    

        # range of omegas to iterate over          
        omegas = np.arange(1, 2, 0.01)
        iterations = np.zeros_like(omegas) # storage array for convergence iterations for omegas

        # iterate over each value of omega and determine how many iterations it take for the system to converge to a solution    
        for i, omega in enumerate(omegas):     

            self.omega = omega 

            # initialise a 2D phi grid (as oppose to 3D with the other algorithms)
            self.phi = np.zeros((self.rows, self.cols))

            # starting error such that the while loops runs
            self.phi_difference = 1000

            # counter for the number of iterations that have passed
            count = 0
            
            # iterate the system until it has converged to a user defined tolerance
            while self.phi_difference > self.tol:

                # save the current value of phi
                copy_phi = np.copy(self.phi)

                # update the system with SOR rules
                self.update_SOR()
                #self.updater

                # difference measure
                self.phi_difference = np.sum(np.abs(self.phi - copy_phi))
                print(self.phi_difference)

                # mark that another iteration has passed
                count += 1
                
                if count > 10000:   # prevents infinite loops                    
                    break

            iterations[i] = count

        # save to file
        print(np.shape(omegas), np.shape(iterations))
        data = np.vstack((omegas, iterations)).T
        print(data)
        np.savetxt(f'omega_convergence_iterations({self.rows}x{self.cols},tol={self.tol}).txt', data)
        
        # plot the number of iterations against omega, save plot
        plt.plot(omegas, iterations)
        plt.ylabel('Number of iterations')
        plt.xlabel('$\omega$')
        #plt.title('')
        plt.savefig(f'omega_convergence_iterations({self.rows}x{self.cols},tol={self.tol}).png')
        plt.show()
            
    # functions related to running the simulation

    def run_simulation(self):
        '''
        The simulation is ran for either electric or magnetic fields and a contour plot of the potential and corresponding
        electric or magnetic fields are plotted.
        '''

        # run algorithm until it has converged (difference measure is less than a tolerance)

        self.phi_difference = 1000

        # Jacobi
        if self.algorithm_type == 'j':
            
            # update phi based on jacobi algorithm while difference measure is less than a tolerance
            while self.phi_difference > self.tol:
                self.update_jacobi() # updates phi grid and the difference measure

            # show and save contour plot of the final potential
            self.plot_phi_contour()

            if self.field == 'E': # point charge
                self.compute_E_field() # get components of the electric field based off of the potential
                self.plot_E_field() # plot vector field of the electric field
            
            if self.field == 'M': # current rod
                self.compute_B_field() # get components of the electric field based off of the potential
                self.plot_B_field() # plot vector field of the electric field

        # Gauss-Seidel
        if self.algorithm_type == 'g':

            # update phi based on Gauss-Seidel algorithm while difference measure is less than a tolerance
            while self.phi_difference > self.tol:
                self.update_GS() # updates phi grid and the difference measure

            # show and save contour plot of the final potential
            self.plot_phi_contour()

            if self.field == 'E': # point charge
                self.compute_E_field() # get components of the electric field based off of the potential
                self.plot_E_field() # plot vector field of the electric field

            if self.field == 'M': # current rod
                self.compute_B_field() # get components of the electric field based off of the potential
                self.plot_B_field() # plot vector field of the electric field

        if self.algorithm_type == 's':
            # look for the best omega value for quickest convergence, plot no iterations vs omega
            self.omega_analysis()


def main():
    '''
    Solve the poisson equation
    '''

    N = int(input('Please input the required system size (cubic) : '))

    tolerance = float(input('What should the accuracy of the system be? ' ))

    field = input('Magnetic or Electric field? (M or E): ')

    algorithm = input('Please choose the algorithm to follow (Jacobi: j, Gauss-Seidel: g or SOR: s): ')

    p = Poisson(N1 = N, N2 = N, N3 = N, tol = tolerance, field = field, algorithm = algorithm)

if __name__ == "__main__":
    main()

