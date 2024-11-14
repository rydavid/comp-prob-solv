import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initialize_lattice(size):
    """
    Initializes a square lattice of the given size, where all sites are initially empty (represented by 0).

    Parameters:
    - size (int): The length of one side of the square lattice.

    Returns:
    - lattice (ndarray): A 2D numpy array with shape (size, size), initialized with zeros.
    """
    # create a lattice with all sites set to 0 (empty)
    lattice = np.zeros([size, size])
    
    return lattice


def compute_neighbor_indices(size):
    """
    Computes the neighboring indices for each site in a square lattice with periodic boundary conditions.

    Parameters:
    - size (int): The length of one side of the square lattice.

    Returns:
    - neighbor_indices (dict): A dictionary where each key is a tuple representing a site (x, y),
                               and the value is a list of tuples representing its neighbors' coordinates.
    """
    # initialize an empty dictionary to store the neighbors for each site
    neighbor_indices = {}
    for x in range(size):
        for y in range(size):
            # calculate the neighboring sites with periodic boundary conditions
            neighbors = [
                ((x - 1) % size, y),  # neighbor to the left
                ((x + 1) % size, y),  # neighbor to the right
                (x, (y - 1) % size),  # neighbor above
                (x, (y + 1) % size)   # neighbor below
            ]
            # assign the list of neighbors to the current site
            neighbor_indices[(x, y)] = neighbors

    return neighbor_indices


def calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB):
    """
    Calculates the interaction energy for a given particle at a specific site in the lattice,
    based on its neighboring particles and their interaction strengths.

    Parameters:
    - lattice (ndarray): The 2D grid representing the system.
    - site (tuple): The coordinates (x, y) of the site where the particle is located.
    - particle (int): The type of particle (1 for A, 2 for B).
    - neighbor_indices (dict): A dictionary containing the neighboring coordinates for each site.
    - epsilon_AA (float): Interaction energy between two particles of type A.
    - epsilon_BB (float): Interaction energy between two particles of type B.
    - epsilon_AB (float): Interaction energy between a particle of type A and a particle of type B.

    Returns:
    - interaction_energy (float): The total interaction energy for the particle at the given site.
    """
    # extract the coordinates of the site
    x, y = site
    interaction_energy = 0
    
    # iterate over each neighboring site
    for neighbor in neighbor_indices[(x, y)]:
        # get the particle at the neighboring site
        neighbor_particle = lattice[neighbor]
        if neighbor_particle != 0:  # if the neighbor is not an empty site
            if particle == 1:  # particle A
                if neighbor_particle == 1:  # neighbor is also particle A
                    interaction_energy += epsilon_AA
                else:  # neighbor is particle B
                    interaction_energy += epsilon_AB
            else:  # particle B
                if neighbor_particle == 2:  # neighbor is also particle B
                    interaction_energy += epsilon_BB
                else:  # neighbor is particle A
                    interaction_energy += epsilon_AB

    return interaction_energy


def attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params):
    """
    Attempts to either add a particle (A or B) to an empty site or remove a particle from an occupied site
    in a lattice, based on a Monte Carlo acceptance probability.

    Parameters:
    - lattice (ndarray): The 2D grid representing the system, where 0 indicates an empty site, 1 indicates particle A, and 2 indicates particle B.
    - N_A (int): The current number of particles of type A in the lattice.
    - N_B (int): The current number of particles of type B in the lattice.
    - N_empty (int): The current number of empty sites in the lattice.
    - neighbor_indices (array-like): The indices representing the neighbors of a site for interaction energy calculations.
    - params (dict): Dictionary containing simulation parameters such as temperature (T), interaction energies (epsilon_A, epsilon_B, etc.), and chemical potentials (mu_A, mu_B).

    Returns:
    - N_A (int): Updated number of particles of type A.
    - N_B (int): Updated number of particles of type B.
    - N_empty (int): Updated number of empty sites.
    """
    # extract information/parameters for logic and calculations
    size = lattice.shape[0]
    N_sites = size * size
    beta = 1 / params['T']
    epsilon_A, epsilon_B, epsilon_AA, epsilon_BB, epsilon_AB, mu_A, mu_B = (
    params['epsilon_A'], params['epsilon_B'], params['epsilon_AA'], 
    params['epsilon_BB'], params['epsilon_AB'], params['mu_A'], params['mu_B']
    )
    
    add = np.random.choice([True, False])
    
    if add:
        # adding a particle
        if N_empty == 0:
            return N_A, N_B, N_empty  # no empty sites available
        
        # find empty sites and select one randomly
        empty_sites = np.argwhere(lattice == 0)
        site = empty_sites[np.random.choice(len(empty_sites))]
        
        # randomly choose whether to add A or B
        A = np.random.choice([True, False])
        if A:
            particle = 1
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:  # adding particle B
            particle = 2
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
        
        # calculate energy change and acceptance probability
        delta_E = epsilon + calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)
        acc_prob = min(1, (N_empty) / (N_s + 1) * np.exp(-beta * (delta_E - mu)))
        
        # perform the move with probability acc_prob
        r = np.random.rand()
        if r < acc_prob:
            lattice[site[0], site[1]] = particle
            if particle == 1:
                N_A += 1
            else:
                N_B += 1
            N_empty -= 1

    else:  # removing a particle
        # check if there are any particles to remove
        if N_sites - N_empty == 0:
            return N_A, N_B, N_empty  # no particles to remove
        
        # find occupied sites and select one randomly
        occupied_sites = np.argwhere(lattice != 0)
        site = occupied_sites[np.random.choice(len(occupied_sites))]
        
        # get the particle at the chosen site
        particle = lattice[site[0], site[1]]
        
        # determine which particle it is
        if particle == 1:
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else: 
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
        
        # calculate energy change and acceptance probability
        delta_E = -epsilon - calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)
        acc_prob = min(1, N_s / (N_empty + 1) * np.exp(-beta * (delta_E + mu)))
        
        # perform the move with probability acc_prob
        r = np.random.rand()
        if r < acc_prob:
            lattice[site[0], site[1]] = 0
            if particle == 1:
                N_A -= 1
            else:
                N_B -= 1
            N_empty += 1
    
    return N_A, N_B, N_empty


def run_simulation(size, n_steps, params):
    """
    Runs a Monte Carlo simulation on a lattice for a given number of steps.

    Parameters:
    - size (int): The length of one side of the square lattice.
    - n_steps (int): The number of simulation steps to run.
    - params (dict): A dictionary containing the simulation parameters, including interaction energies and chemical potentials.

    Returns:
    - lattice (ndarray): The final state of the lattice after the simulation.
    - coverage_A (ndarray): An array tracking the coverage of particle A over time (fraction of lattice occupied by A).
    - coverage_B (ndarray): An array tracking the coverage of particle B over time (fraction of lattice occupied by B).
    """
    # initialize the lattice and compute the neighbor indices for periodic boundary conditions
    lattice = initialize_lattice(size)
    neighbor_indices = compute_neighbor_indices(size)
    
    # calculate the total number of sites on the lattice
    N_sites = size * size
    N_A = 0  # initial number of particle A
    N_B = 0  # initial number of particle B
    N_empty = N_sites  # initially, all sites are empty
    
    # initialize arrays to store coverage information for particles A and B at each step
    coverage_A = np.full(n_steps, np.nan)
    coverage_B = np.full(n_steps, np.nan)

    # run the simulation for the specified number of steps
    for step in range(n_steps):
        # attempt to add or remove a particle and update particle counts
        N_A, N_B, N_empty = attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params)
        
        # record the current coverage of particles A and B
        coverage_A[step] = N_A / N_sites
        coverage_B[step] = N_B / N_sites

    return lattice, coverage_A, coverage_B


def plot_lattice(lattice, ax, title):
    """
    Plots the current state of the lattice using matplotlib, displaying particles of type A and B in different colors.

    Parameters:
    - lattice (ndarray): The 2D grid representing the system.
    - ax (matplotlib axis): The axis on which to plot the lattice.
    - title (str): The title for the plot.

    Returns:
    - ax (matplotlib axis): The axis with the plotted lattice.
    """
    # get the size of the lattice
    size = lattice.shape[0]
    
    # iterate over each site in the lattice
    for x in range(size):
        for y in range(size):
            if lattice[x, y] == 1:  # if site contains particle A
                ax.plot(x + 0.5, y + 0.5, 'o', color='red')
            elif lattice[x, y] == 2:  # if site contains particle B
                ax.plot(x + 0.5, y + 0.5, 'o', color='blue')
    
    # set axis limits, ticks, and gridlines
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(np.arange(0, size + 1, 1))
    ax.set_yticks(np.arange(0, size + 1, 1))
    
    # remove axis labels for a cleaner plot
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # add gridlines for site boundaries
    ax.grid(which='major', linestyle='-', color='black', linewidth=0.5)
    
    # set the title of the plot
    ax.set_title(title)
    
    return ax


def animate_simulation(size, n_steps, params_list, length=60000):
    """
    Animate the adsorption process for a series of mu_A and T values, running the Monte Carlo simulation
    step-by-step for each set of parameters.

    Args:
        size (int): The size of the lattice (size x size grid).
        n_steps (int): Number of Monte Carlo steps to run for each animation.
        params_list (list): List of dictionaries containing simulation parameters. Each dictionary should
                            have the keys 'mu_A', 'mu_B', 'T', 'epsilon_A', 'epsilon_B', 'epsilon_AA',
                            'epsilon_BB', 'epsilon_AB' for each set of parameters.
        length (int): The duration of the animation in milliseconds

    Returns:
        ani: The matplotlib animation object.
    """
    
    # initialize the figure and axis
    fig, ax = plt.subplots()
    
    # initialize lattice and simulation variables
    lattice = initialize_lattice(size)
    neighbor_indices = compute_neighbor_indices(size)
    N_sites = size * size
    N_A = 0
    N_B = 0
    N_empty = N_sites

    # create empty arrays for coverage tracking
    coverage_A = np.full(n_steps, np.nan)
    coverage_B = np.full(n_steps, np.nan)

    # set the axis limits and other plot properties
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(np.arange(0, size + 1, 1))
    ax.set_yticks(np.arange(0, size + 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='major', linestyle='-', color='black', linewidth=0.5)

    # parameter index to keep track of the current parameter set
    param_index = [0]

    # update function for the animation
    def update(frame):
        """
        Update the plot for the current frame and parameter set.
        
        Args:
            frame (int): The current step/frame in the animation.
        """
        nonlocal N_A, N_B, N_empty, lattice, neighbor_indices

        # reset the lattice if we finish simulating one set of parameters
        if frame % n_steps == 0 and frame > 0:
            param_index[0] += 1  # move to the next parameter set
            if param_index[0] >= len(params_list):
                param_index[0] = 0  # restart from the first parameter set
            lattice = initialize_lattice(size)
            N_A = 0
            N_B = 0
            N_empty = N_sites

        # get the current set of parameters
        params = params_list[param_index[0]]
        mu_A = params['mu_A']
        T = params['T']

        # run 100 steps of the simulation (update lattice)
        for _ in range(1000):  # only update the plot every 100 steps
            N_A, N_B, N_empty = attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params)
        
        # store the current coverage for A and B
        step = frame % n_steps
        coverage_A[step] = N_A / (size * size)
        coverage_B[step] = N_B / (size * size)
        
        # clear the axis and plot the updated lattice
        ax.cla()
        plot_lattice(lattice, ax, f"$\mu_H$: {mu_A:.2f}, T: {T:.4f} - Step: {step}\nCoverage H: {coverage_A[step]:.2f}, Coverage N: {coverage_B[step]:.2f}")
    
    # scale the speed of the animation to the number of steps
    interval = length / n_steps

    # create the animation object
    ani = animation.FuncAnimation(fig, update, np.arange(0, n_steps * len(params_list), 1000), interval=interval, repeat=False)
    
    # animation.Animation.save(ani, 'simulation.mp4')

    # plt.show()

    return ani

# ideal mixture
# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)
params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0,
            'epsilon_BB': 0,
            'epsilon_AB': 0,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k)
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    print()
    lattice, coverage_A, coverage_B = run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
    mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5))

fig.suptitle('Ideal Mixture of Nitrogen and Hydrogen')

# Mean coverage of A
axs[0].pcolormesh(mus_A, Ts, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$')
axs[0].set_ylabel(r'$T$')

# Mean coverage of B
axs[1].pcolormesh(mus_A, Ts, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_H + \theta_N \rangle$')
axs[2].set_xlabel(r'$\mu_H$')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

# Plot the final lattice configuration

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = plot_lattice(final_lattice[0, 3], axs[3], f'$\mu_H = -0.2$ eV,\n$T = 0.01 / k$')

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = plot_lattice(final_lattice[3, 3], axs[4], f'$\mu_H = -0.1$ eV,\n$T = 0.01 / k$')

# mu_A = 0 eV and T = 0.01 / k
axs[5] = plot_lattice(final_lattice[6, 3], axs[5], f'$\mu_H = 0$ eV,\n$T = 0.01 / k$')

plt.tight_layout()
plt.savefig('ideal', dpi=300)

animation.Animation.save(animate_simulation(size, n_steps, params), 'ideal.mp4', dpi=50, fps=3, bitrate=500)

# repuslsive regime
# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)
params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0.05,
            'epsilon_BB': 0.05,
            'epsilon_AB': 0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k)
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    print()
    lattice, coverage_A, coverage_B = run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
    mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5))

fig.suptitle('Repulsive Interactions between Nitrogen and Hydrogen')

# Mean coverage of A
axs[0].pcolormesh(mus_A, Ts, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$')
axs[0].set_ylabel(r'$T$')

# Mean coverage of B
axs[1].pcolormesh(mus_A, Ts, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_H + \theta_N \rangle$')
axs[2].set_xlabel(r'$\mu_H$')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

# Plot the final lattice configuration

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = plot_lattice(final_lattice[0, 3], axs[3], f'$\mu_H = -0.2$ eV,\n$T = 0.01 / k$')

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = plot_lattice(final_lattice[3, 3], axs[4], f'$\mu_H = -0.1$ eV,\n$T = 0.01 / k$')

# mu_A = 0 eV and T = 0.01 / k
axs[5] = plot_lattice(final_lattice[6, 3], axs[5], f'$\mu_H = 0$ eV,\n$T = 0.01 / k$')

plt.tight_layout()
plt.savefig('repulsive', dpi=300)

animation.Animation.save(animate_simulation(size, n_steps, params), 'repulsive.mp4', dpi=50, fps=3, bitrate=500)

# attractive regime
# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)
params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': -0.05,
            'epsilon_BB': -0.05,
            'epsilon_AB': -0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k)
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    print()
    lattice, coverage_A, coverage_B = run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
    mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5))

fig.suptitle('Attractive Interactions between Nitrogen and Hydrogen')

# Mean coverage of A
axs[0].pcolormesh(mus_A, Ts, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$')
axs[0].set_ylabel(r'$T$')

# Mean coverage of B
axs[1].pcolormesh(mus_A, Ts, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_H + \theta_N \rangle$')
axs[2].set_xlabel(r'$\mu_H$')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

# Plot the final lattice configuration

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = plot_lattice(final_lattice[0, 3], axs[3], f'$\mu_H = -0.2$ eV,\n$T = 0.01 / k$')

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = plot_lattice(final_lattice[3, 3], axs[4], f'$\mu_H = -0.1$ eV,\n$T = 0.01 / k$')

# mu_A = 0 eV and T = 0.01 / k
axs[5] = plot_lattice(final_lattice[6, 3], axs[5], f'$\mu_H = 0$ eV,\n$T = 0.01 / k$')

plt.tight_layout()
plt.savefig('attractive', dpi=300)

animation.Animation.save(animate_simulation(size, n_steps, params), 'attractive.mp4', dpi=50, fps=3, bitrate=500)

# immiscible particles
# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)
params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': -0.05,
            'epsilon_BB': -0.05,
            'epsilon_AB': 0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k)
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    print()
    lattice, coverage_A, coverage_B = run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
    mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5))

fig.suptitle('Immiscible Hydrogen and Nitrogen')

# Mean coverage of A
axs[0].pcolormesh(mus_A, Ts, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$')
axs[0].set_ylabel(r'$T$')

# Mean coverage of B
axs[1].pcolormesh(mus_A, Ts, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_H + \theta_N \rangle$')
axs[2].set_xlabel(r'$\mu_H$')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

# Plot the final lattice configuration

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = plot_lattice(final_lattice[0, 3], axs[3], f'$\mu_H = -0.2$ eV,\n$T = 0.01 / k$')

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = plot_lattice(final_lattice[3, 3], axs[4], f'$\mu_H = -0.1$ eV,\n$T = 0.01 / k$')

# mu_A = 0 eV and T = 0.01 / k
axs[5] = plot_lattice(final_lattice[6, 3], axs[5], f'$\mu_H = 0$ eV,\n$T = 0.01 / k$')

plt.tight_layout()
plt.savefig('immiscible', dpi=300)

animation.Animation.save(animate_simulation(size, n_steps, params), 'immiscible.mp4', dpi=50, fps=3, bitrate=500)

# like dissolves unlike
# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)
params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0.05,
            'epsilon_BB': 0.05,
            'epsilon_AB': -0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k)
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    print()
    lattice, coverage_A, coverage_B = run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
    mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5))

fig.suptitle("Like Dissolves Unlike Scenario")

# Mean coverage of A
axs[0].pcolormesh(mus_A, Ts, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$')
axs[0].set_ylabel(r'$T$')

# Mean coverage of B
axs[1].pcolormesh(mus_A, Ts, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_H + \theta_N \rangle$')
axs[2].set_xlabel(r'$\mu_H$')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

# Plot the final lattice configuration

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = plot_lattice(final_lattice[0, 3], axs[3], f'$\mu_H = -0.2$ eV,\n$T = 0.01 / k$')

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = plot_lattice(final_lattice[3, 3], axs[4], f'$\mu_H = -0.1$ eV,\n$T = 0.01 / k$')

# mu_A = 0 eV and T = 0.01 / k
axs[5] = plot_lattice(final_lattice[6, 3], axs[5], f'$\mu_H = 0$ eV,\n$T = 0.01 / k$')

plt.tight_layout()
plt.savefig('odd', dpi=300)

animation.Animation.save(animate_simulation(size, n_steps, params), 'odd.mp4', dpi=50, fps=3, bitrate=500)