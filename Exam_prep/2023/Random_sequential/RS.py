
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
        
        # generate lattice from rows and cols, assigning values of 0 (R), 1 (P), 2 (S) to each cell
        self.grid = np.random.choice(a = [0, 1, 2], size = [self.rows, self.rows])

        # number of times to sweep over the grid
        self.sweeps = sweeps

        # initialise global probabilities
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        # #randomly assign values of 0 (R), 1 (P), 2 (S) to each cell
        # for i in range(self.rows):
        #     for j in range(self.cols):

        #         self.grid[i, j] = np.random.choice([0, 1, 2])   

        # # initialise the grid randomly
        # if initialisation == 'random':

        #     # randomly assign values of 0 (S), 1 (I), 2 (R) to each cell
        #     for i in range(self.rows):
        #         for j in range(self.cols):

        #             self.grid[i, j] = np.random.choice([0, 1, 2])    

        # # modified grid so that some agents are immune to infection
        # elif initialisation == 'vaccinated':

        #     # initialise random grid
        #     for i in range(self.rows):
        #         for j in range(self.cols):

        #             self.grid[i, j] = np.random.choice([0, 1, 2]) 

        #     # vaccinate a fraction of the cells, i.e set value to 4 so the rules arent applied to them
        #     self.vaccinate_cells(vaccinated_fraction)
                    
        # animate if requested   
        if type == 'animation':
            self.animate_SIRS()

        # else:
        #     self.plot_minority_fraction()
            
        # elif type == 'phase_diagram':
        #     self.phase_diagram()

        # elif type == 'immune_fraction':
        #     self.measure_immunity_fraction()


    # def count_infected_neighbours(self, i, j):
    #     '''
    #     Find the number of infected neighbours around a given cell ij. Neighbours go as this:
    #        n2
    #     n1 ij n3  
    #        n4

    #     Parameters:

    #     i: integer
    #         row value of cell
    #     j: integer 
    #         column value of cell

    #     returns:
        
    #     infected_neighbours: integer
    #         Number of neighbours that are infected around cell i, j

    #     changes: could use np.row as a convolution
    #     '''

    #     # get status of each neighbour
    #     # n1 = self.grid[(i-1)%self.rows, (j)%self.cols]
    #     # n2 = self.grid[(i)%self.rows, (j+1)%self.cols]
    #     # n3 = self.grid[(i+1)%self.rows, (j)%self.cols]
    #     # n4 = self.grid[(i)%self.rows, (j-1)%self.cols]

    #     n1 = self.grid[(i-1)%self.rows, (j+1)%self.cols]
    #     n2 = self.grid[i, (j+1)%self.cols]
    #     n3 = self.grid[(i+1)%self.rows, (j+1)%self.cols]
    #     n4 = self.grid[(i-1)%self.rows, j]
    #     n5 = self.grid[(i+1)%self.rows, j]
    #     n6 = self.grid[(i-1)%self.rows, (j-1)%self.cols]
    #     n7 = self.grid[i, (j-1)%self.cols]
    #     n8 = self.grid[(i+1)%self.rows, (j-1)%self.cols]

    #     neighbours = np.array([n1, n2, n3, n4, n5, n6, n7, n8])

    #     # count how many neighbours are infected
    #     infected_neighbours = np.count_nonzero(neighbours == 1)

    #     return infected_neighbours
    
    def sum_state_neighbours(self, i, j, state):
        '''
        Counts the number of neighbours of a cell in a specific state

        state:
            0 = Rock
            1 = Paper
            2 = Scissors

        neighbours are indexed as follows:
        n1  n2  n3
        n4  ij  n5
        n6  n7  n8
        '''

        # value of each neighbour
        n1 = self.grid[(i-1)%self.rows, (j+1)%self.cols]
        n2 = self.grid[i, (j+1)%self.cols]
        n3 = self.grid[(i+1)%self.rows, (j+1)%self.cols]
        n4 = self.grid[(i-1)%self.rows, j]
        n5 = self.grid[(i+1)%self.rows, j]
        n6 = self.grid[(i-1)%self.rows, (j-1)%self.cols]
        n7 = self.grid[i, (j-1)%self.cols]
        n8 = self.grid[(i+1)%self.rows, (j-1)%self.cols]

        # array containing all of the neighbours
        neighbours = np.array([n1, n2, n3, n4, n5, n6, n7, n8])

        count = np.count_nonzero(neighbours == state)

        return count


    def apply_rules(self, i, j):
        '''
        Apply the SIRS rules to a single cell in the grid. Update is sequential

        Parameters:

        i: integer
            row value of cell
        j: integer 
            column value of cell
        '''

        # current status of cell, 0 = R, 1 = P, 2 = S
        status = self.grid[i, j]

        # cell is currently rock
        if status == 0:

            # count how many neighbours are paper
            paper_neighbour_count = self.sum_state_neighbours(i, j, 1)

            # if one or more neighbours are infected, there is a probability p1 that the cell will infect
            if paper_neighbour_count >= 1:
                
                # probability
                r = np.random.uniform(0, 1) # generate random number between 0 and 1
                
                # infect the cell if random number is less than p1
                if r < self.p1:
                    self.grid[i, j] = 1

        # if cell is paper
        elif status == 1:
            
            # how many neighbours are in the scissors state around i, j
            scissors_neighbour_count = self.sum_state_neighbours(i, j, 2)

            # if one or more neighbours are infected, there is a probability p1 that the cell will infect
            if scissors_neighbour_count >= 1:
                
                # probability
                r = np.random.uniform(0, 1) # generate random number between 0 and 1
                
                # infect the cell if random number is less than p1
                if r < self.p2:
                    self.grid[i, j] = 2

        # if cell is scissors
        elif status == 2:
            
            # how many neighbours are in the scissors state around i, j
            scissors_neighbour_count = self.sum_state_neighbours(i, j, 0)

            # if one or more neighbours are infected, there is a probability p1 that the cell will infect
            if scissors_neighbour_count >= 1:
                
                # probability
                r = np.random.uniform(0, 1) # generate random number between 0 and 1
                
                # infect the cell if random number is less than p1
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


    def minority_state(self, equilibrium_sweeps = 100, sweeps = 1000):
        '''
        Measure the minority fraction as a function of p3, once the system has reached a steady state after a defined number
        of equilibrium sweeps
        '''

        fractions = []

        # begin simulation
        for k in range(sweeps):
                    
            # update the grid with specified rules
            self.evolve_infection()

            # system has been allowed to reach steady state
            if k >= equilibrium_sweeps:

                # find minority fraction (measure number of R, P, S and find which is smallest)
                R = np.count_nonzero(self.grid == 0)
                P = np.count_nonzero(self.grid == 1)
                S = np.count_nonzero(self.grid == 2)

                # find smallest between R, P and S. Get minority fraction from the smallest
                fraction = np.min(np.array([R, P, S]))/(self.rows*self.cols)
                #print(np.min(np.array([R, P, S])), fraction, R + P + S, R, P, S)

                # store fraction
                fractions.append(fraction)

        average_fraction = np.sum(np.array(fractions))/len(fractions)
        variance = np.var(fractions)

        return average_fraction, variance
    

    def plot_minority_fraction(self, p3_min = 0, p3_max = 0.1, p3_resolution = 0.01):
        '''
        Run model for different values of p3, plotting the average minority fraction for each
        '''
        # range of p3 to iterate over
        p3 = np.arange(p3_min, p3_max + p3_resolution, p3_resolution)

        average_fraction = np.zeros_like(p3)
        variance = np.zeros_like(p3)

        for i in tqdm(range(len(p3))):

            average_fraction[i], variance[i] = self.minority_state(p3[i], equilibrium_sweeps = 100, sweeps = 1000)

        # save the data
        combined_array = np.column_stack((p3, average_fraction, variance))
        np.savetxt(f'p3_between_{p3_min}&{p3_max}).csv', combined_array, delimiter = ';')

        # plot and save
        plt.scatter(p3, average_fraction, s = 5)
        plt.xlabel('p3')
        plt.ylabel('Average minority fraction')
        plt.savefig(f'p3_between_{p3_min}&{p3_max}).png')

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



def main():

    # run sirs
    sirs = SIRS(p1 = 0.5, p2 = 0.5, p3 = 0.1, N1 = 50, N2 = 50)


if __name__ == "__main__":
    main()

