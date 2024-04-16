'''
Plotting
'''

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Glauber_Data.txt', sep = " ", header = 1)
data.columns = ['T', '<E>', 'E_error', '<|M|>', 'c', 'c_error', 'Chi']

print(data)

# plot energy
plt.errorbar(data['T'], data['<E>'], yerr = data['E_error'], fmt='ro', markersize = 2)
plt.xlabel("Temperature, $T$")
plt.ylabel("Average energy, $\langle E \\rangle$")
plt.savefig('Energy.png')
plt.show()

# # plot specific heat
# plt.errorbar(T, c , yerr=c_err, fmt='ro')
# plt.xlabel("Temperature, $T$")
# plt.ylabel("Scaled Heat Capacity, $c$")
# plt.savefig('ScaledHeatCapacity.png')
# plt.show()

# # plot magnetisation 
# plt.scatter(T, M)
# plt.xlabel("Temperature, $T$")
# plt.ylabel("$\langle |M| \\rangle$")
# plt.savefig('Magnetisation.png')
# plt.show()

# # plot susceptibility
# plt.scatter(T, chi)
# plt.xlabel("Temperature, T")
# plt.ylabel("Susceptibility, $\chi$")
# plt.savefig('Susceptibility.png')
# plt.show()