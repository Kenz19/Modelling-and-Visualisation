
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# for animation
import matplotlib as mpl
mpl.use('TKAgg')


class SIRS():

    def __init__(self, p1, p2, p3, vaccinated_fraction = 0.1, N1 = 50, N2 = 50, initialisation = 'random', type = 'animation', sweeps = 1000):
        '''
        Inialisation function for SIRS model class, initialises
        the lattice grid and the probabilities for infection are input
        into self

        p1: float,
            probability of infection (S -> I)
        p2: float, 
            probability or recovery (I -> R)
        p3: float, 
            probability of becoming susceptible (R -> S)
        N1: integer, 
            number of rows in lattice
        N2: integer, 
            number of columns in lattice
        '''

        # lattice dimensions
        self.rows = int(N1) # number of rows in lattice
        self.cols = int(N2) # number of columns in lattice
        
        # generate lattice from rows and cols
        self.grid = np.zeros([self.rows, self.rows])

        # number of times to sweep over the grid
        self.sweeps = sweeps

        # initialise global probabilities
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        # initialise the grid randomly
        if initialisation == 'random':

            # randomly assign values of 0 (S), 1 (I), 2 (R) to each cell
            for i in range(self.rows):
                for j in range(self.cols):

                    self.grid[i, j] = np.random.choice([0, 1, 2])    

        # modified grid so that some agents are immune to infection
        elif initialisation == 'vaccinated':

            # initialise random grid
            for i in range(self.rows):
                for j in range(self.cols):

                    self.grid[i, j] = np.random.choice([0, 1, 2]) 

            # vaccinate a fraction of the cells, i.e set value to 4 so the rules arent applied to them
            self.vaccinate_cells(vaccinated_fraction)
                    
        # animate if requested   
        if type == 'animation':
            self.animate_SIRS()
            
        elif type == 'phase_diagram':
            self.phase_diagram()

        elif type == 'immune_fraction':
            self.measure_immunity_fraction()


    def count_infected_neighbours(self, i, j):
        '''
        Find the number of infected neighbours around a given cell ij. Neighbours go as this:
           n2
        n1 ij n3  
           n4

        Parameters:

        i: integer
            row value of cell
        j: integer 
            column value of cell

        returns:
        
        infected_neighbours: integer
            Number of neighbours that are infected around cell i, j

        changes: could use np.row as a convolution
        '''

        # get status of each neighbour
        n1 = self.grid[(i-1)%self.rows, (j)%self.cols]
        n2 = self.grid[(i)%self.rows, (j+1)%self.cols]
        n3 = self.grid[(i+1)%self.rows, (j)%self.cols]
        n4 = self.grid[(i)%self.rows, (j-1)%self.cols]

        neighbours = np.array([n1, n2, n3, n4])

        # count how many neighbours are infected
        infected_neighbours = np.count_nonzero(neighbours == 1)

        return infected_neighbours


    def apply_rules(self, i, j):
        '''
        Apply the SIRS rules to a single cell in the grid

        Parameters:

        i: integer
            row value of cell
        j: integer 
            column value of cell
        '''

        # current status of cell, 0 = S, 1 = I, 2 = R
        status = self.grid[i, j]

        # cell is currently susceptible
        if status == 0:

            # check number of neighbours that are infected
            infected_neighbour_count = self.count_infected_neighbours(i, j)

            # if one or more neighbours are infected, there is a probability p1 that the cell will infect
            if infected_neighbour_count >= 1:
                
                # probability
                r = np.random.uniform(0, 1) # generate random number between 0 and 1
                
                # infect the cell if random number is less than p1
                if r < self.p1:
                    self.grid[i, j] = 1

        # if cell is infected
        elif status == 1:

            # probability
            r = np.random.uniform(0, 1) # generate random number between 0 and 1

            # cell recovers from infection if r less than p2
            if r < self.p2:
                self.grid[i, j] = 2

        # if cell is in recovery
        elif status == 2:

            # probability
            r = np.random.uniform(0, 1) # generate random number between 0 and 1

            # cell becomes susceptible
            if r < self.p3:
                self.grid[i, j] = 0

    def evolve_infection(self):
        '''
        Perform a single sweep of the SIRS model
        '''

        # randomly update the number of cells equivalent to the count in the grid
        for i in range(self.rows*self.cols):

            # randomly choose a cell to update on the grid
            i, j = np.random.randint(0, self.rows), np.random.randint(0, self.cols) 

            # update random lattice position based on the rules of the SIRS model
            #print('old ' + str(self.grid[i, j]))
            self.apply_rules(i, j)

            #print('new ' + str(self.grid[i, j]))


    def animate_SIRS(self):
        '''
        Animate the SIRS model
        '''

        fig = plt.figure()

        for i in range(self.sweeps):

            # perform sweeps 
            self.evolve_infection()

            # visualise every 10th state even outside of equilibrium
            if i % 5 == 0:
                
                plt.cla()
                im = plt.imshow(self.grid, animated=True)
                plt.draw()
                plt.pause(0.0001)

    # functions for measurements ##

    @staticmethod
    def measure_cell_type(grid, cell_type):
        '''
        lattice: numpy array

        Cell type: string
            S = susceptible, I = infected, R = recovered

        returns:

        count: integer
            Number of cells of specified type on grid
        '''

        # count how many susceptible cells are present in the lattice
        if cell_type == 'S':
            count = np.count_nonzero(grid == 0)

        # count how many infected cells are present in the lattice
        elif cell_type == 'I':
            count = np.count_nonzero(grid == 1)

        # count how many recovered cells are present in the lattice
        elif cell_type == 'R':
            count = np.count_nonzero(grid == 2)

        return count
    
    @staticmethod
    def phase_diagram(upperp1 = 0.5, lowerp1 = 0.2, upperp3 = 0.5, lowerp3 = 0.5, p2 = 0.5, resolution = 0.02, sweeps = 10000, no_equilibration_sweeps = 100, p3 = 'const'):
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

        # generate list of p1 and p3 values based on specified resolution
        p1_vals = np.arange(lowerp1, upperp1 + resolution, resolution)
        no_p1 = len(p1_vals)
        print(no_p1)
        p3_vals = np.arange(lowerp3, upperp3 + resolution, resolution)
        no_p3 = len(p3_vals)

        # if only 1 p3 value is to be looped over, set to constant
        if p3 == 'const':
            p3_vals = np.array([0.5])
            no_p3 = len(p3_vals)

        # array to store average infected cells
        I = np.zeros((no_p1, no_p3))
        I2 = np.zeros((no_p1, no_p3)) # sqaure of I
        Ierr = np.zeros((no_p1, no_p3)) # errors on I

        # loop over p1 values
        for i in tqdm(range(no_p1)):
            # loop over p3 values

            for j in tqdm(range(no_p3)):

                # initialise SIRS grid
                sirs = SIRS(p1 = p1_vals[i], p2 = p2, p3 = p3_vals[j], type = None, sweeps = sweeps)

                #array for number of infected cells
                no_infected = np.zeros(sweeps - no_equilibration_sweeps) 
                no_infected2 = np.zeros(sweeps - no_equilibration_sweeps)
                #print('number of infected cells = ' + str(len(no_infected)))

                # begin simulation
                for k in range(sweeps):
                    
                    # update the sirs grid
                    sirs.evolve_infection()

                    # if system has reached equilibrium
                    if k >= no_equilibration_sweeps:

                        # measure the number of infected cells
                        no_infected[k-no_equilibration_sweeps] = sirs.measure_cell_type(sirs.grid, 'I')
                        no_infected2[k-no_equilibration_sweeps] = (sirs.measure_cell_type(sirs.grid, 'I'))**2

                # output the mean infected measurement for this p1 and p3 combination
                I[i, j] = np.mean(no_infected)
                I2[i, j] = np.mean(no_infected2)
                Ierr[i, j] = sirs.bootstrap_errors(no_infected, sirs.rows, sirs.cols, 1000)

                #print(I, I2, Ierr)
        #print(I, I2, Ierr)

        # # output the average number of infected cells
        np.savetxt("Checkpoint_2/SIRS/phase.csv", I, delimiter=",")
        np.savetxt("Checkpoint_2/SIRS/phase2.csv", I2, delimiter=",")
        np.savetxt("Checkpoint_2/SIRS/phase_err.csv", Ierr, delimiter =",")

        # save output
        #np.savetxt("Checkpoint_2/SIRS/phase.csv", (I,I2,Ierr))

        # dict = {'<I>': I, '<I2>': I2, 'Ierr': Ierr}

        # data = pd.DataFrame.from_dict(dict, orient='columns')#, dtype=None, columns=None)

        # data.to_csv("cut.csv", index = False)


    def vaccinate_cells(self, fraction):
        '''
        Vaccinate a specified fraction of cells in global grid

        fraction: float
            Value between 0 and 1 to be vaccinated in grid
        '''

        # number of cells to vaccinate
        no_vaccinations = int(fraction*self.rows*self.cols)

        for i in range(no_vaccinations):
            
            # choose random cell to vaccinate
            i, j = np.random.randint(0, self.rows), np.random.randint(0, self.cols) 

            # ensure that randomly chosen site is not already vaccinated, if it is, choose another lattice site 
            while self.grid[i, j] == 4: # 4 = vaccinated
                i, j = np.random.randint(0, self.rows), np.random.randint(0, self.cols) 

            self.grid[i, j] = 4

    @staticmethod
    def measure_immunity_fraction(upper_fraction = 1, lower_fraction = 0, step = 0.05, sweeps = 1000, no_equilibration_sweeps = 100):
        '''
        Measure the number of immune cells as a function of the fraction of immune cells

        upper_fraction: float
            Highest fraction of immune cells to be initialised in grid

        lower_fraction: float
            Lowest fraction of immune cells to be initialised in grid

        steps: float
            Step size between upper_fraction and lower_fraction

        sweeps: integer
            Number of sweeps to carry out per simulation

        no_equilibration_sweeps: integer
            Number of sweeps to pass before measurements occur.
        '''

        # fraction of immune cells
        frac = np.arange(lower_fraction, upper_fraction + step, step)
        no_frac = len(frac)

        I = np.zeros(no_frac)

        # loop over immunity fractions
        for i in tqdm(range(no_frac)):

            # initialise SIRS grid with 
            sirs = SIRS(p1 = 0.5, p2 = 0.5, p3 = 0.5, initialisation = 'vaccinated', vaccinated_fraction = frac[i], type = None)

            #array for number of infected cells
            no_infected = np.zeros(sweeps - no_equilibration_sweeps)

            # begin simulation
            for k in range(sweeps):
                
                # update the sirs grid
                sirs.evolve_infection()

                # if system has reached equilibrium
                if k >= no_equilibration_sweeps:

                    # measure the number of infected cells
                    no_infected[k-no_equilibration_sweeps] = sirs.measure_cell_type(sirs.grid, 'I')
            
            # output the mean infected measurement for this p1 and p3 combination
            I[i] = np.mean(no_infected)

        infected = I/(sirs.rows*sirs.cols)

        dict = {'Immune_fraction': frac, '<I>': infected}

        data = pd.DataFrame.from_dict(dict, orient='columns')#, dtype=None, columns=None)

        # output the average number of infected cells
        data.to_csv("infected_vs_fraction.csv", index = False)
        

    @staticmethod
    def bootstrap_errors(data, rows, cols, no_k):
        '''
        Calculate errors via the bootstrap method (general function for c and suseptibility calculations)

        Parameters
        ----------
        data : array
            data for error to be calculated for.
        rows : integer
            Number of rows in grid.
        cols : integer
            Number of columns in grid.
        no_k : integer
            Number of iterations in the bootstrap method

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
            value = (np.mean(resample**2)-np.mean(resample)**2)/(rows*cols)
            values.append(value)
        
        # error calculation from the randomly sampled data
        values = np.array(values)
        error = np.sqrt(np.mean(values**2)-np.mean(values)**2)
        
        return error


def main():

    # system inputs
    N = float(input('System size: ' )) # system size

    # probabilities
    p1 = float(input('p1: '))
    p2 = float(input('p2: '))
    p3 = float(input('p3: '))

    # animation, phase diagram, immunity
    sirs_type = input('type of sim (animation, phase_diagram, immune_fraction): ')

    # run sirs
    sirs = SIRS(p1 = p1, p2 = p2, p3 = p3, N1 = N, N2 = N, type = sirs_type)

    


    #sirs.phase_diagram()

    # 0.5, 0.5, 0.5 = dynamics equilibrium (active phase)
    
    # to go to absorbing phase, decrease value of p1 (like social distancing), or set to 0.5, 0.5, 0.05

    # wave behaviour is in prozimity to transition

    #print(sirs.grid)

if __name__ == "__main__":
    main()

