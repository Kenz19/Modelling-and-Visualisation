from RS import SIRS
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

def plot_minority_fraction(p3_min = 0, p3_max = 0.1, p3_resolution = 0.005):
        '''
        Run model for different values of p3, plotting the average minority fraction for each
        '''
        # range of p3 to iterate over
        p3 = np.arange(p3_min, p3_max + p3_resolution, p3_resolution)

        average_fraction = np.zeros_like(p3)
        variance = np.zeros_like(p3)

        no_sweeps = 1000

        for i in tqdm.tqdm(range(len(p3))):

            # initialise sirs class
            sirs = SIRS(p1 = 0.5, p2 = 0.5, p3 = p3[i], vaccinated_fraction = 0.1, N1 = 50, N2 = 50, initialisation = 'random', type = 'else', sweeps = no_sweeps)

            average_fraction[i], variance[i] = sirs.minority_state(equilibrium_sweeps = 100, sweeps = no_sweeps)
            #print(average_fraction)

        # save the data
        combined_array = np.column_stack((p3, average_fraction, variance))
        np.savetxt(f'p3_between_{p3_min}&{p3_max}.csv', combined_array, delimiter = ';')
        print(average_fraction)

        # plot and save
        plt.scatter(p3, average_fraction, s = 5)
        #plt.plot(p3, average_fraction)
        plt.xlabel('p3')
        plt.ylabel('Average minority fraction')
        plt.savefig(f'p3_between_{p3_min}&{p3_max}.png')
        plt.show()

        plt.scatter(p3, variance, s = 5)
        plt.xlabel('p3')
        plt.ylabel('Average minority fraction variance')
        plt.savefig(f'p3_between_{p3_min}&{p3_max}-variance.png')
        plt.show()


plot_minority_fraction()