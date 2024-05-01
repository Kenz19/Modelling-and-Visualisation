import numpy as np
import matplotlib.pyplot as plt

def update_cell(current_state, neighbor_states, p1, p2, p3):
    # Update rules
    if current_state == 'R' and 'P' in neighbor_states:
        if np.random.random() < p1:
            return 'P'
    elif current_state == 'P' and 'S' in neighbor_states:
        if np.random.random() < p2:
            return 'S'
    elif current_state == 'S' and 'R' in neighbor_states:
        if np.random.random() < p3:
            return 'R'
    return current_state

def update_lattice(lattice, p1, p2, p3):
    N = lattice.shape[0]
    new_lattice = np.empty_like(lattice)
    
    for i in range(N):
        for j in range(N):
            current_state = lattice[i, j]
            neighbor_states = [
                lattice[(i-1)%N, (j-1)%N], lattice[(i-1)%N, j], lattice[(i-1)%N, (j+1)%N],
                lattice[i, (j-1)%N], lattice[i, (j+1)%N],
                lattice[(i+1)%N, (j-1)%N], lattice[(i+1)%N, j], lattice[(i+1)%N, (j+1)%N]
            ]
            new_lattice[i, j] = update_cell(current_state, neighbor_states, p1, p2, p3)
    
    return new_lattice

def count_minority_phase(lattice):
    counts = np.unique(lattice, return_counts=True)[1]
    return np.min(counts) / np.sum(counts)

def simulate(N, p1, p2, p3, num_steps=1000, num_trials=10):
    minority_phase_avg = np.zeros_like(p3)
    minority_phase_var = np.zeros_like(p3)
    
    for idx, p3_val in enumerate(p3):
        minority_phase_samples = np.zeros(num_trials)
        for trial in range(num_trials):
            lattice = np.random.choice(['R', 'P', 'S'], size=(N, N))
            for step in range(num_steps):
                lattice = update_lattice(lattice, p1, p2, p3_val)
            minority_phase_samples[trial] = count_minority_phase(lattice)
        minority_phase_avg[idx] = np.mean(minority_phase_samples)
        minority_phase_var[idx] = np.var(minority_phase_samples)
    
    return minority_phase_avg, minority_phase_var

# Parameters
N = 50
p1 = 0.5
p2 = 0.5
p3_values = np.linspace(0, 0.1, 11)  # p3 values from 0 to 0.1 with 0.01 resolution

# Simulation
avg, var = simulate(N, p1, p2, p3_values)

# Plot
plt.errorbar(p3_values, avg, yerr=np.sqrt(var), fmt='-o')
plt.xlabel('p3')
plt.ylabel('Average fraction of minority phase')
plt.title('Average Fraction of Minority Phase vs p3')
plt.grid(True)
plt.show()