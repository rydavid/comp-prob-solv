import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cnsts

def initialize_chain(n_particles, box_size, r0):
    """
    Initializes the positions of a polymer chain in a cubic box with periodic boundary conditions.

    Parameters:
        n_particles (int): Number of particles in the chain.
        box_size (float): Size of the simulation box.
        r0 (float): Equilibrium bond length.

    Returns:
        np.ndarray: Array of shape (n_particles, 3) containing particle positions.
    """
    positions = np.zeros((n_particles, 3))
    current_position = np.array([box_size / 2, box_size / 2, box_size / 2])
    positions[0] = current_position
    for i in range(1, n_particles):
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)  # Normalize to unit vector
        next_position = current_position + r0 * direction
        positions[i] = apply_pbc(next_position, box_size)
        current_position = positions[i]
    return positions

def initialize_velocities(n_particles, target_temperature, mass):
    """
    Initializes the velocities of particles from the Maxwell-Boltzmann distribution.

    Parameters:
        n_particles (int): Number of particles.
        target_temperature (float): Target temperature.
        mass (float): Mass of each particle.

    Returns:
        np.ndarray: Array of shape (n_particles, 3) containing particle velocities.
    """
    velocities = np.random.normal(0, np.sqrt(target_temperature / mass), (n_particles, 3))
    velocities -= np.mean(velocities, axis=0)  # Remove net momentum
    return velocities

def apply_pbc(position, box_size):
    """
    Applies periodic boundary conditions to a position.

    Parameters:
        position (np.ndarray): Position vector.
        box_size (float): Size of the simulation box.

    Returns:
        np.ndarray: Position vector after applying periodic boundary conditions.
    """
    return position % box_size

def compute_harmonic_forces(positions, k, r0, box_size):
    """
    Computes harmonic forces between consecutive particles in the chain.

    Parameters:
        positions (np.ndarray): Array of particle positions.
        k (float): Spring constant.
        r0 (float): Equilibrium bond length.
        box_size (float): Size of the simulation box.

    Returns:
        np.ndarray: Array of forces on each particle.
    """
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    for i in range(n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = displacement - box_size * np.round(displacement / box_size)  # Minimum image convention
        distance = np.linalg.norm(displacement)
        force_magnitude = -k * (distance - r0)
        force = force_magnitude * (displacement / distance)
        forces[i] -= force
        forces[i + 1] += force
    return forces

def compute_lennard_jones_forces(positions, epsilon, sigma, box_size, interaction_type):
    """
    Computes Lennard-Jones forces between particles in the chain.

    Parameters:
        positions (np.ndarray): Array of particle positions.
        epsilon (float): Depth of the Lennard-Jones potential well.
        sigma (float): Finite distance where the potential is zero.
        box_size (float): Size of the simulation box.
        interaction_type (str): Interaction type ('repulsive' or 'attractive').

    Returns:
        np.ndarray: Array of forces on each particle.
    """
    n_particles = len(positions)
    forces = np.zeros_like(positions)
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            if interaction_type == 'repulsive' and abs(i - j) == 2:
                epsilon_current = epsilon
            elif interaction_type == 'attractive' and abs(i - j) > 2:
                epsilon_current = epsilon
            else:
                continue
            displacement = positions[j] - positions[i]
            displacement = displacement - box_size * np.round(displacement / box_size)  # Minimum image convention
            distance = np.linalg.norm(displacement)
            if distance < (sigma * 2**(1/6)):
                force_magnitude = 24 * epsilon_current * ((sigma / distance)**12 - 0.5 * (sigma / distance)**6) / distance
                force = force_magnitude * (displacement / distance)
                forces[i] -= force
                forces[j] += force
    return forces

def velocity_verlet(positions, velocities, forces, dt, mass):
    """
    Integrate equations of motion using the velocity Verlet algorithm.

    Parameters:
        positions (np.ndarray): Array of particle positions.
        velocities (np.ndarray): Array of particle velocities.
        forces (np.ndarray): Array of forces on each particle.
        dt (float): Time step.
        mass (float): Mass of each particle.

    Returns:
        tuple: Updated positions, velocities, and forces.
    """
    velocities += 0.5 * forces / mass * dt
    positions += velocities * dt
    positions = apply_pbc(positions, box_size)
    forces_new = compute_harmonic_forces(positions, k, r0, box_size) + compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive') + compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'attractive')
    velocities += 0.5 * forces_new / mass * dt
    return positions, velocities, forces_new

def rescale_velocities(velocities, target_temperature, mass):
    """
    Rescales velocities to match the target temperature.

    Parameters:
        velocities (np.ndarray): Array of particle velocities.
        target_temperature (float): Target temperature.
        mass (float): Mass of each particle.

    Returns:
        np.ndarray: Rescaled velocities.
    """
    kinetic_energy = 0.5 * mass * np.sum(np.linalg.norm(velocities, axis=1)**2)
    current_temperature = (2 / 3) * kinetic_energy / ((len(velocities)) * cnsts.k)
    scaling_factor = np.sqrt(target_temperature / current_temperature)
    velocities *= scaling_factor
    return velocities

def calculate_radius_of_gyration(positions):
    """
    Calculates the radius of gyration of the polymer chain.

    Parameters:
        positions (np.ndarray): Array of particle positions.

    Returns:
        float: Radius of gyration.
    """
    center_of_mass = np.mean(positions, axis=0)
    Rg_squared = np.mean(np.sum((positions - center_of_mass)**2, axis=1))
    Rg = np.sqrt(Rg_squared)
    return Rg

def calculate_end_to_end_distance(positions):
    """
    Calculates the end-to-end distance of the polymer chain.

    Parameters:
        positions (np.ndarray): Array of particle positions.

    Returns:
        float: End-to-end distance.
    """
    Ree = np.linalg.norm(positions[-1] - positions[0])
    return Ree

def compute_harmonic_potential_energy(positions, k, r0, box_size):
    """
    Computes the harmonic potential energy of the polymer chain.

    Parameters:
        positions (np.ndarray): Array of particle positions.
        k (float): Spring constant.
        r0 (float): Equilibrium bond length.
        box_size (float): Size of the simulation box.

    Returns:
        float: Total harmonic potential energy.
    """
    n_particles = len(positions)
    potential_energy = 0.0
    for i in range(n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = displacement - box_size * np.round(displacement / box_size)  # Minimum image convention
        distance = np.linalg.norm(displacement)
        potential_energy += 0.5 * k * (distance - r0)**2
    return potential_energy

def compute_lennard_jones_potential(positions, epsilon, sigma, box_size, interaction_type):
    """
    Computes Lennard-Jones potential energy between particles in the chain.

    Parameters:
        positions (np.ndarray): Array of particle positions.
        epsilon (float): Depth of the Lennard-Jones potential well.
        sigma (float): Finite distance where the potential is zero.
        box_size (float): Size of the simulation box.
        interaction_type (str): Interaction type ('repulsive' or 'attractive').

    Returns:
        float: Lennard-Jones potential energy.
    """
    n_particles = len(positions)
    potential_energy = 0.0
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            displacement = positions[j] - positions[i]
            displacement = displacement - box_size * np.round(displacement / box_size)  # Minimum image convention
            distance = np.linalg.norm(displacement)
            if distance < (sigma * 2**(1/6)):
                if interaction_type == 'repulsive' and abs(i - j) == 2:
                    epsilon_current = epsilon
                    potential_energy += 4 * epsilon_current * ((sigma / distance)**12 - (sigma / distance)**6 + 0.25)
                elif interaction_type == 'attractive' and abs(i - j) > 2:
                    epsilon_current = epsilon
                    potential_energy += 4 * epsilon_current * ((sigma / distance)**12 - (sigma / distance)**6)
                else:
                    continue
            else:
                if interaction_type == 'repulsive' and abs(i - j) == 2:
                    epsilon_current = epsilon
                    potential_energy += 0
                elif interaction_type == 'attractive' and abs(i - j) > 2:
                    epsilon_current = epsilon
                    potential_energy += 4 * epsilon_current * ((sigma / distance)**12 - (sigma / distance)**6)
                else:
                    continue
    return potential_energy

np.random.seed(42)

# Simulation parameters
dt = 0.01  # Time step
total_steps = 20000  # Number of steps
box_size = 100.0  # Size of the cubic box
k = 0.1  # Spring constant
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
target_temperature = 0.1  # Target temperature
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_repulsive = 0.1  # Depth of repulsive LJ potential
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # LJ potential parameter

# Arrays to store properties
temperatures = np.linspace(0.1, 1.0, 10)
Rg_values = []
Ree_values = []
potential_energies = []
potential_energy_array = []
positions_array = []

for T in temperatures:
    # Set target temperature
    target_temperature = T
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, target_temperature, mass)
    # Simulation loop
    for step in range(total_steps):
        # Compute forces and potentials
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive
        
        # Integrate equations of motion
        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)
        
        # Apply thermostat
        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, target_temperature, mass)

    for step in range(2000):
        # Compute forces and potentials
        forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
        forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
        total_forces = forces_harmonic + forces_repulsive + forces_attractive
        potential_harmonic = compute_harmonic_potential_energy(positions, k, r0, box_size)
        potential_repulsive = compute_lennard_jones_potential(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
        potential_attractive = compute_lennard_jones_potential(positions, epsilon_attractive, sigma, box_size, 'attractive')
        potential_energy = potential_attractive + potential_harmonic + potential_repulsive
        potential_energy_array.append(potential_energy)
        
        # Integrate equations of motion
        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        positions_array.append(positions)
        
        # Apply thermostat
        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, target_temperature, mass)


    positions = np.mean(np.array(positions_array), axis = 0)
    
    # Compute properties
    Rg = calculate_radius_of_gyration(positions)
    Ree = calculate_end_to_end_distance(positions)
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(np.mean(potential_energy_array))

# # Plotting
# plt.figure()
# plt.plot(temperatures, Rg_values, label='Radius of Gyration')
# plt.xlabel('Temperature')
# plt.ylabel('Radius of Gyration')
# plt.title('Radius of Gyration vs Temperature')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(temperatures, Ree_values, label='End-to-End Distance')
# plt.xlabel('Temperature')
# plt.ylabel('End-to-End Distance')
# plt.title('End-to-End Distance vs Temperature')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(temperatures, potential_energies, label='Potential Energy')
# plt.xlabel('Temperature')
# plt.ylabel('Potential Energy')
# plt.title('Potential Energy vs Temperature')
# plt.legend()
# plt.show()

# k_vals = np.linspace(0.1, 2, 20)
# for k in k_vals:
#     # Set target temperature
#     target_temperature = 0.3
#     positions = initialize_chain(n_particles, box_size, r0)
#     velocities = initialize_velocities(n_particles, target_temperature, mass)
#     # Simulation loop
#     for step in range(total_steps):
#         # Compute forces and potentials
#         forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
#         forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
#         forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
#         total_forces = forces_harmonic + forces_repulsive + forces_attractive
        
#         # Integrate equations of motion
#         positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)
        
#         # Apply thermostat
#         if step % rescale_interval == 0:
#             velocities = rescale_velocities(velocities, target_temperature, mass)

#     for step in range(2000):
#         # Compute forces and potentials
#         forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
#         forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
#         forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
#         total_forces = forces_harmonic + forces_repulsive + forces_attractive
#         potential_harmonic = compute_harmonic_potential_energy(positions, k, r0, box_size)
#         potential_repulsive = compute_lennard_jones_potential(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
#         potential_attractive = compute_lennard_jones_potential(positions, epsilon_attractive, sigma, box_size, 'attractive')
#         potential_energy = potential_attractive + potential_harmonic + potential_repulsive
#         potential_energy_array.append(potential_energy)
        
#         # Integrate equations of motion
#         positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

#         positions_array.append(positions)
        
#         # Apply thermostat
#         if step % rescale_interval == 0:
#             velocities = rescale_velocities(velocities, target_temperature, mass)


#     positions = np.mean(np.array(positions_array), axis = 0)
    
#     # Compute properties
#     Rg = calculate_radius_of_gyration(positions)
#     Ree = calculate_end_to_end_distance(positions)
#     Rg_values.append(Rg)
#     Ree_values.append(Ree)
#     potential_energies.append(np.mean(potential_energy_array))

# # Plotting
# plt.figure()
# plt.plot(k_vals, Rg_values, label='Radius of Gyration')
# plt.xlabel('k')
# plt.ylabel('Radius of Gyration')
# plt.title('Radius of Gyration vs k')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(k_vals, Ree_values, label='End-to-End Distance')
# plt.xlabel('k')
# plt.ylabel('End-to-End Distance')
# plt.title('End-to-End Distance vs k')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(k_vals, potential_energies, label='Potential Energy')
# plt.xlabel('k')
# plt.ylabel('Potential Energy')
# plt.title('Potential Energy vs k')
# plt.legend()
# plt.show()

# epsilon_repulsives = np.linspace(0.1, 2, 20)
# for epsilon_repulsive in epsilon_repulsives:
#     # Set target temperature
#     target_temperature = 0.3
#     k=0.1
#     positions = initialize_chain(n_particles, box_size, r0)
#     velocities = initialize_velocities(n_particles, target_temperature, mass)
#     # Simulation loop
#     for step in range(total_steps):
#         # Compute forces and potentials
#         forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
#         forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
#         forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
#         total_forces = forces_harmonic + forces_repulsive + forces_attractive
        
#         # Integrate equations of motion
#         positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)
        
#         # Apply thermostat
#         if step % rescale_interval == 0:
#             velocities = rescale_velocities(velocities, target_temperature, mass)

#     for step in range(2000):
#         # Compute forces and potentials
#         forces_harmonic = compute_harmonic_forces(positions, k, r0, box_size)
#         forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
#         forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size, 'attractive')
#         total_forces = forces_harmonic + forces_repulsive + forces_attractive
#         potential_harmonic = compute_harmonic_potential_energy(positions, k, r0, box_size)
#         potential_repulsive = compute_lennard_jones_potential(positions, epsilon_repulsive, sigma, box_size, 'repulsive')
#         potential_attractive = compute_lennard_jones_potential(positions, epsilon_attractive, sigma, box_size, 'attractive')
#         potential_energy = potential_attractive + potential_harmonic + potential_repulsive
#         potential_energy_array.append(potential_energy)
        
#         # Integrate equations of motion
#         positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

#         positions_array.append(positions)
        
#         # Apply thermostat
#         if step % rescale_interval == 0:
#             velocities = rescale_velocities(velocities, target_temperature, mass)


#     positions = np.mean(np.array(positions_array), axis = 0)
    
#     # Compute properties
#     Rg = calculate_radius_of_gyration(positions)
#     Ree = calculate_end_to_end_distance(positions)
#     Rg_values.append(Rg)
#     Ree_values.append(Ree)
#     potential_energies.append(np.mean(potential_energy_array))

# # Plotting
# plt.figure()
# plt.plot(epsilon_repulsives, Rg_values, label='Radius of Gyration')
# plt.xlabel(r'$\epsilon_\text{rep}$')
# plt.ylabel('Radius of Gyration')
# plt.title(r'Radius of Gyration vs $\epsilon_\mathrm{rep}$')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(epsilon_repulsives, Ree_values, label='End-to-End Distance')
# plt.xlabel(r'$\epsilon_\text{rep}$')
# plt.ylabel('End-to-End Distance')
# plt.title(r'End-to-End Distance vs $\epsilon_\mathrm{rep}$')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(epsilon_repulsives, potential_energies, label='Potential Energy')
# plt.xlabel(r'$\epsilon_\text{rep}$')
# plt.ylabel('Potential Energy')
# plt.title(r'Potential Energy vs $\epsilon_\text{rep}$')
# plt.legend()
# plt.show()