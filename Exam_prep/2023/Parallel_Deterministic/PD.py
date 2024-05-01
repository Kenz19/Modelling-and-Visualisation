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

    def __init__(self, N1 = 50, N2 = 50, animation = 'no', iterations = 1000): # initialisation
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
            Animate the game of life or not, either yes or no (default = yes). If animation = yes, the game will evolve
            and the state will be displayed in real time. If no, measurements will be carried out.

        Returns
        -------
        lattice : Numpy array
            Lattice containing spin up and spin down units - spins randomised
        '''

        self.rows = int(N1) # number of rows in lattice
        self.cols = int(N2) # number of columns in lattice
        
        # generate lattice from rows and cols
        self.grid = np.zeros([self.rows, self.cols])

        # initialise the lattice as the pie wedge formation
        self.pie_wedge_initialisation()

        self.iterations = iterations



        # initalise lattice randomly
        # if initialisation == 'random':

        #     p = 0.5 # probability threshold for alive (1) or dead (0) cell.
            
        #     # assign each lattice point 0 or 1
        #     for i in range(N1):
        #         for j in range(N2):
        #             r = random.random() # generate random number for each position in the lattice (between 0 and 1)
        #             if r < p: self.grid[i, j] = 0 # dead cell
        #             if r >= p: self.grid[i, j] = 1 # alive cell

        # # initialise glider in empty lattice
        # elif initialisation == 'glider':

        #     # pick random point for centre of glider
        #     i, j = random.randint(0, self.rows), random.randint(0, self.cols)

        #     # set up glider
        #     self.grid[(i-1)%self.rows, j%self.cols] = 1
        #     self.grid[i%self.rows, (j+1)%self.cols] = 1
        #     self.grid[(i+1)%self.rows, (j-1)%self.cols] = 1
        #     self.grid[(i+1)%self.rows, j%self.cols] = 1
        #     self.grid[(i+1)%self.rows, (j+1)%self.cols] = 1
        #     #print(i, j, (i+1)%self.rows, (j+1)%self.cols)

        # elif initialisation == 'blinker':
        #     '''
        #     set up blinker state
        #     '''
        #     i,j = random.randint(0, self.rows), random.randint(0, self.cols)

        #     # set up blinker
        #     self.grid[i%self.rows, (j+1)%self.cols] = 1
        #     self.grid[i%self.rows, j%self.cols] = 1
        #     self.grid[i%self.rows, (j-1)%self.cols] = 1   

        #plt.imshow(self.grid)

        # animate if called for
        if animation == 'yes':
            self.show_the_game()

        # # otherwise, run the game of life without animation
        else: 
            self.track_state()


    def pie_wedge_initialisation(self):
        '''
        Take in an N1 x N2 grid and initialise it with a pie wedge formation
        '''

        # segments = np.split(self.grid, 3, axis=0)

        # segments[0].fill(0)  # First segment (zeros)
        # segments[1].fill(1)  # Second segment (ones)
        # segments[2].fill(0)

        # plt.imshow(segments)

        # import numpy as 

        # Define angles in radians
        theta1 = np.deg2rad(0)  # Start angle for segment 1
        theta2 = np.deg2rad(120)  # Start angle for segment 2
        theta3 = np.deg2rad(240)  # Start angle for segment 3
        theta4 = np.deg2rad(360)

        # Define the center of the array
        center_x, center_y = int(self.rows/2), int(self.cols//2)

        # Define the radius (half of the array size)
        radius = int(self.rows/2)

        # Fill segments based on angles
        for i in range(self.rows):
            for j in range(self.cols):
                x = i - center_x
                y = j - center_y
                angle = np.arctan2(y, x)  # Calculate angle from center to (i, j) - returns between -pi ad pi

                angle = (angle + 2*np.pi) % (2*np.pi)

                # greater or equal than 0 and less than 120 degrees
                if angle >= theta1 and angle < theta2:

                    self.grid[i, j] = 0 # set to rock

                elif angle >= theta2 and angle < theta3:

                    self.grid[i, j] = 1 # set to paper

                elif angle >= theta3 and angle < theta4:

                    self.grid[i, j] = 2 # set to rock

        # plt.imshow(self.grid)
        # plt.colorbar()
        # plt.show()
    

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

    def evolve(self):
        '''
        Evolve the game of life for 1 iteration based on the rules of the game:

        • A cell in the R state with more than two (i.e., 3 or more out of 8) neighbours in the P state changes its state to P.
        • A cell in the P state with more than two S neighbours becomes S.
        • A cell in the S state with more than two R neighbours becomes R.

        This updates the current version of the grid
        '''

        # grid of cells where value = number of neighbouring cells that are alive
        temp_grid = np.ones([self.rows, self.cols]) # 

        # iterate over entire grid (each cell)
        for i in range(self.rows): # for each row
            for j in range(self.cols): # for each column
                
                # cells that are rock
                if self.grid[i, j] == 0:
                    pN = self.sum_state_neighbours(i, j, 1) # number of neighbours that are paper

                    # if the number of paper neighbours is larger than 2 then the rock becomes paper
                    if pN > 2:
                        temp_grid[i, j] = 1
                    
                    # rock remains rock if insufficient neighbours
                    else:
                        temp_grid[i, j] = 0

                # cells that are currently paper
                elif self.grid[i, j] == 1:  
                    pS = self.sum_state_neighbours(i, j, 2) # number of neighbours that are scissors

                    # change to scissors
                    if pS > 2:
                        temp_grid[i, j] = 2
                    
                    # remain as paper otherwise
                    else:
                        temp_grid[i, j] = 1   

                # cells that are currently scissors
                elif self.grid[i, j] == 2:   
                    pR = self.sum_state_neighbours(i, j, 0) # number of rock neighbours
                    
                    # transform to rock
                    if pR > 2:
                        temp_grid[i, j] = 0

                    # remain as scissors
                    else:
                        temp_grid[i, j] = 2
                        
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

    def track_state(self):
        '''
        Measure the state value for a single point on an updating grid, i, j
        '''

        # fixed grid point to track state of
        i = int(self.rows/2)
        j = int(self.cols/2)

        state = np.zeros(self.iterations)
        time = np.zeros(self.iterations)

        #state = []
        #time = []

        for k in tqdm(range(self.iterations)):
            
            # evolve the grid
            self.evolve()

            # record the current state of the grid
            state[k] = self.grid[i, j]

            # record time
            time[k] = k + 1

        # save the state and time data
        combined_array = np.column_stack((time, state))
        np.savetxt(f'State_vals(i={i},j={j},{self.rows}x{self.cols}).csv', combined_array, delimiter = ';')

        # plot and save 
        plt.scatter(time, state, s = 1)
        plt.xlim(300,400)
        plt.ylabel('State')
        plt.xlabel('Time')
        plt.title(f'i = {i} j = {j}. R = 0, P = 1, S = 2')
        plt.savefig(f'State_vals(i={i},j={j},{self.rows}x{self.cols}).png')
        plt.show()






def main():

    # get desired system initialisation
    #initialistation = input('How should the system be initalised? (random, blinker, glider): ')

    N = int(input('What is the system size? '))

    game = GameOfLife(N1 = N, N2 = N)
    game.pie_wedge_initialisation()


        #measure = input('Measure the equilibrium time? (yes or no): ')

        # # print equilibrium time to terminal if want to measure
        # if measure == 'yes':
        #    game.measure_equilibrium_time()

    # elif initialistation == 'blinker':
    #     game = GameOfLife(initialisation = 'blinker')

    # elif initialistation == 'glider':
    #     game = GameOfLife(initialisation = 'glider')
    
    # if initialisation was invalid, tell user and run main again for another input
    # else:
    #     print('Invalid initialisation, please input one of the following: random, blinker or glider.')
    #     main()



if __name__ == "__main__":
    main()
