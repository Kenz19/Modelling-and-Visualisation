'''
Checkpoint 2 - game of life

Trying the class method as a learning experience
'''

### imports

# modules
import matplotlib as mpl
mpl.use('TKAgg')

import numpy as np
import random
import matplotlib.pyplot as plt

from tqdm import tqdm

###

class GameOfLife():

    def __init__(self, N1 = 50, N2 = 50, initialisation = 'random', animation = 'yes', iterations = 1000):
        '''
        Initialise the Game of life. A cell with value 1 is alive, a cell with value 0 is dead.
    
        Parameters
        ----------
        N1 : Integer
            Number of rows in the game of life grid (default = 50)
        N2 : Integer
            Number of columns in the game of life grid (default = 50)
        initialisation : string
            Initial state of the system, either random, blinker, glider (default = random)
        animation : string
            Animate the game of life or not, either yes or no (default = yes)

        Returns
        -------
        lattice : Numpy array
            Lattice containing spin up and spin down units - spins randomised
        '''

        self.rows = int(N1) # number of rows in lattice
        self.cols = int(N2) # number of columns in lattice
        
        # generate lattice from rows and cols
        self.grid = np.zeros([self.rows, self.cols])

        self.iterations = iterations

        # initalise lattice randomly
        if initialisation == 'random':

            p = 0.5 # probability threshold for alive (1) or dead (0) cell.
            
            # assign each lattice point 0 or 1
            for i in range(N1):
                for j in range(N2):
                    r = random.random() # generate random number for each position in the lattice (between 0 and 1)
                    if r < p: self.grid[i, j] = 0 # dead cell
                    if r >= p: self.grid[i, j] = 1 # alive cell

        # initialise glider in empty lattice
        elif initialisation == 'glider':

            # pick random point for centre of glider
            i, j = random.randint(0, self.rows), random.randint(0, self.cols)

            # set up glider
            self.grid[(i-1)%self.rows, j%self.cols] = 1
            self.grid[i%self.rows, (j+1)%self.cols] = 1
            self.grid[(i+1)%self.rows, (j-1)%self.cols] = 1
            self.grid[(i+1)%self.rows, j%self.cols] = 1
            self.grid[(i+1)%self.rows, (j+1)%self.cols] = 1
            #print(i, j, (i+1)%self.rows, (j+1)%self.cols)

        elif initialisation == 'blinker':
            '''
            set up blinker state
            '''
            i,j = random.randint(0, self.rows), random.randint(0, self.cols)

            # set up blinker
            self.grid[i%self.rows, (j+1)%self.cols] = 1
            self.grid[i%self.rows, j%self.cols] = 1
            self.grid[i%self.rows, (j-1)%self.cols] = 1   

        #plt.imshow(self.grid)

        # animate if called for
        if animation == 'yes':
            self.show_the_game()

        # # otherwise, run the game of life without animation
        # else: 
        #     self.play_the_game()


    def find_alive_neighbours(self, i, j):
        '''
        Find the number of neighbours that are alive. 

        Periodic boundry conditions applied

        labelling for neighbours around cell ij is as follows:

        n1  n2  n3
        n4  ij  n5
        n6  n7  n8

        '''

        # track number of cells alive around each neighbour
        live_count = 0

        # identify each neighbour, boundry conditions present if cell ij on boundry
        n1 = self.grid[(i-1)%self.rows, (j+1)%self.cols]
        n2 = self.grid[i, (j+1)%self.cols]
        n3 = self.grid[(i+1)%self.rows, (j+1)%self.cols]
        n4 = self.grid[(i-1)%self.rows, j]
        n5 = self.grid[(i+1)%self.rows, j]
        n6 = self.grid[(i-1)%self.rows, (j-1)%self.cols]
        n7 = self.grid[i, (j-1)%self.cols]
        n8 = self.grid[(i+1)%self.rows, (j-1)%self.cols]

        # count number of live neighbours (1 = alive, 0 = dead)
        live_count = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8

        return live_count

        
    def evolve(self):
        '''
        Evolve the game of life for 1 iteration based on the rules of the game:

        • Any live cell with less than 2 live neighbours dies.
        • Any live cell with 2 or 3 live neighbours lives on to the next step.
        • Any live cell with more than 3 live neighbours dies.
        • Any dead cell with exactly 3 live neighbours becomes alive.

        This updates the current version of the grid
        '''

        # grid of cells where value = number of neighbouring cells that are alive
        temp_grid = np.ones([self.rows, self.cols]) # 

        # iterate over entire grid (each cell)
        for i in range(self.rows): # for each row
            for j in range(self.cols): # for each column

                # count the number of live neighbours to each cell
                alive_neighbour_count = self.find_alive_neighbours(i, j)

                # apply rules of the game applicable to currently live cells
                if self.grid[i, j] == 1:

                    if alive_neighbour_count < 2 or alive_neighbour_count > 3: # remove float
                        temp_grid[i, j] = 0

                # apply rules of game applicable to currently dead cells
                elif self.grid[i, j] == 0:

                    # if number of live neighbours = 3, cell ressurects
                    if alive_neighbour_count == 3:
                        temp_grid[i, j] = 1

                    # if cell remains dead, communicate this to the temp state
                    else: temp_grid[i, j] = 0             
        
        # set global grid to new state
        self.grid = temp_grid


    def play_the_game(self):#, iterations = self.iterations):
        '''
        Runs the game of life, no animation

        iterations = number of times grid is to be updated
        '''
        # update grid for a number of iterations
        for i in tqdm(range(self.iterations)):
            self.evolve()


    def show_the_game(self):#, iterations = 1000):
        '''Animate the game of life for a given number of frames (iterations)'''

        # initialise animations figure
        fig = plt.figure()

        # update grid for a given number of configurations
        for i in range(self.iterations):

            # update grid
            self.evolve()

            # animate grid as updates happen
            plt.cla()
            im = plt.imshow(self.grid, animated=True)
            plt.draw()
            plt.pause(0.01)

    @staticmethod
    def measure_equilibrium_time(threshold = 10, max_time = 10000):
        '''
        Measure the equilibrium time for each sweep in the game
        '''

        # # initialise game of life
        game = GameOfLife(animation = 'no')

        # count how many are alive on initial grid
        initial_alive = np.sum(game.grid)

        # check how many iterations have passed            
        time = 0 # keep track of time
        no_same_count = 0 # count how many times the number of live cells has remained the same

        # evolve system until it has reached an equilibrium threshold or it has surpassed the maximum time
        while (no_same_count < threshold) and (time < max_time):

            # keep count of iterations
            time += 1

            # update the grid
            game.evolve()

            # measure how many are cells alive in new grid
            current_alive = np.sum(game.grid)
            
            # check if the number of alive cells has remained the same
            if current_alive == initial_alive:
                # track that number of live cells has stayed the same
                no_same_count += 1

            else:
                # assign current live count to inital live count
                initial_alive = current_alive

                # reset consecutive counter
                no_same_count = 0

        # return time once loop has stopped
                
        # if loop reached max time, simulation did not equilibrate therefore return nan
        if time == max_time:
            return np.nan
        
        # system did equilibrate therefore return the time
        else: 
            return time


    # check the next two measurements
    @staticmethod 
    def measure_centre_of_mass(grid):
        '''
        Measure the centre of mass of the live cells in a single grid
        '''

        N1, N2 = grid.shape

        # get coordinates of live cells
        live_coords = np.argwhere(grid)

        # if glider at edge of grid, return nan, as we want to ignore this measurement
        if np.any(live_coords == 0) or np.any(live_coords == N1) or np.any(live_coords == N2):
            return np.array([np.nan, np.nan])
        
        # calculate centre of mass if not at the edge of the grid
        else:
            
            # centre of mass coordinates, sum x and y
            com = np.sum(live_coords, axis = 0)

            # number of coordinates
            n_coords = live_coords.shape[0]

            #print(com/n_coords)
            return com/n_coords
        
    @staticmethod
    def measure_centre_of_mass_velocity(initialisation = 'glider', animation = 'no', iterations = 500):
        """
        Calculate centre of mass of any grid, automatic is glider
        """

        # # initialise game of life
        game = GameOfLife(initialisation = initialisation, animation = animation, iterations = iterations)

        # storage array for centre of mass values
        com = np.zeros([game.iterations, 2])
        iteration = np.zeros([game.iterations])
        #print(com)

        # measure initial centre of mass
        com[0] = game.measure_centre_of_mass(game.grid)
        iteration[0] = 1

        # update grid for a number of iterations, measure centre of mass each time
        for i in tqdm(range(game.iterations)):

            # skip initial measurement as this has been done already
            if i != 0:

                # evolve the grid
                game.evolve()

                # measure new centre of mass
                com[i] = game.measure_centre_of_mass(game.grid)
                iteration[i] = i + 1

        # remove nan values from com
        iteration = iteration[~np.isnan(com).any(axis=1)]
        com = com[~np.isnan(com).any(axis=1)]

        # change in position
        delta_com = np.diff(com, axis = 0) # get difference in positions

        # find magnitudes of change in position
        delta_com_magnitudes = np.linalg.norm(delta_com, axis = 1)
        
        # omit any jumps greater than 5, i.e any that pass the boundry conditions and dont trigger the condition in game.measure_centre_of_mass
        ind = [i for i,v in enumerate(delta_com_magnitudes) if v < 5]

        # get x and y values that are not at the boundries
        xavg = np.mean(delta_com[:, 0][ind])
        yavg = np.mean(delta_com[:, 1][ind])

        velocity = np.sqrt(xavg**2 + yavg**2) # cells/iteration

        return velocity


def main():

    # get desired system initialisation
    initialistation = input('How should the system be initalised? (random, blinker, glider): ')

    # run game based on initialisation input
    if initialistation == 'random':
        game = GameOfLife(initialisation = 'random')

        #measure = input('Measure the equilibrium time? (yes or no): ')

        # # print equilibrium time to terminal if want to measure
        # if measure == 'yes':
        #    game.measure_equilibrium_time()

    elif initialistation == 'blinker':
        game = GameOfLife(initialisation = 'blinker')

    elif initialistation == 'glider':
        game = GameOfLife(initialisation = 'glider')
    
    # if initialisation was invalid, tell user and run main again for another input
    else:
        print('Invalid initialisation, please input one of the following: random, blinker or glider.')
        main()



if __name__ == "__main__":
    main()
