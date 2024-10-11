from scipy.integrate import trapezoid
import scipy
import numpy as np
import matplotlib.pyplot as plt

def lennard_jones(r, epsilon=0.01, sigma=3.4):
    """
    Calculates the Lennard Jones potential using the Lennard Jones formula.
    Is set up for argon by default.

    Parameters:
    r (float): Distance between two atoms in angstroms.
    epsilon (float, optional): Depth of the potential well (i.e., the minimum energy).
    sigma (float, optional): Distance at which the potential is zero.

    Returns:
    float: Calculated Lennard Jones potential.
    """
    V = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    
    return V


def lennard_jones_pf(T, epsilon=0.0103, sigma=3.4, V=10e3):
    """
    Calculate the partition function Z for two Lennard-Jones particles in a cubic box using trapezoidal integration.
    
    Parameters:
    - T (float): Temperature in Kelvin.
    - epsilon (float): Depth of the Lennard-Jones potential well (default 0.0103 angstroms).
    - sigma (float): Characteristic distance at which the Lennard-Jones potential is zero (default 3.4 angstroms).
    - V (float): Volume of the system (default 10e3 angstroms).
    
    Returns:
    - (float): Partition function.
    """
    # define or calculate constants
    k = scipy.constants.k / scipy.constants.eV  # boltzmann constant in eV/K
    k_J = scipy.constants.k
    h = scipy.constants.h  # planck's constant
    NA = scipy.constants.N_A  # avogadro's number
    beta = 1 / (T * k_J)
    m = 39.95 / (1000 * NA)  # mass of argon in kg
    lamb = np.sqrt((beta * h ** 2) / (2 * np.pi * m))
    const = 1 / (lamb ** 6)
    beta = 1 / (T * k)  # redefine beta in eV-1

    # set up grid for integration
    grid_points = 10
    x = np.linspace(-V**(1/3)/2, V**(1/3)/2, grid_points)
    y = np.linspace(-V**(1/3)/2, V**(1/3)/2, grid_points)
    z = np.linspace(-V**(1/3)/2, V**(1/3)/2, grid_points)

    # create meshgrids for each coordinate
    x1, y1, z1 = np.meshgrid(x, y, z, indexing='ij')

    # compute the pairwise distance assuming one particle is at the origin
    r = np.sqrt(x1**2 + y1**2 + z1**2)
    
    # avoid division by zero
    r[r == 0] = 1e-10

    # calculate the values for integration
    gauss = np.exp(-beta * lennard_jones(r, epsilon=epsilon, sigma=sigma))

    # integrate as shamelessly as possible
    integral_x = trapezoid(gauss, x)  # x integration
    integral_y = trapezoid(integral_x, y)    # y integration
    integral_z = trapezoid(integral_y, z)

    # compute volume element
    dx = x[1] - x[0]

    # compute the partition function
    Z = const * integral_z * 2 * dx**3

    return Z


def probability(T, V, Z):
    """
    A function for testing
    """
    k = scipy.constants.k / scipy.constants.eV
    beta = 1 / (T * k)
    gauss = np.exp(-beta * V)
    prob = gauss / Z
    return prob


def calculate_C_v(Z_fn, T):
    """
    Calculate the heat capacity at constant volume (C_v) for a system.

    This function computes the heat capacity at constant volume using the 
    partition function Z as a function of temperature T. The calculations
    are based on the statistical mechanics definition of heat capacity.

    Parameters:
    - Z_fn (function): A function that takes a temperature value as input and returns the 
                       corresponding partition function Z for that temperature.
    - T (array-like): An array of temperature values (in Kelvin) for which to calculate C_v.

    Returns:
    (numpy.ndarray): An array containing the calculated heat capacity at constant volume 
                     for each temperature in T, in units of energy per temperature (eV/K).
    """
    k = scipy.constants.k / scipy.constants.eV
    Z = np.array([Z_fn(T_val) for T_val in T])
    betas = 1 / (k * T)
    ln_Z = np.log(Z)
    U = np.gradient(-ln_Z, betas)
    C_v = np.gradient(U, T)
    return C_v

# set up temperature array
T = np.linspace(1, 200, 100000)

# calculate the heat capacity at constant temperature for each temp in T
C_v = calculate_C_v(lennard_jones_pf, T)

# save as .csv
labels = 'Temperature (K), C_v (eV/K)'
cols = np.column_stack((T, C_v))
np.savetxt('Cv_vs_T.csv', cols, delimiter=',', header=labels, comments='', fmt='%.6e')
