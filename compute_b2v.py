import numpy as np
import scipy
import inspect

def hard_sphere(r, sigma=3.4):
    """
    Calculate the intermolecular potential for a hard-sphere interaction model.

    Parameters:
    r (float): The distance between two particles in angstroms.
    sigma (float, optional): The diameter of the hard sphere (default is 3.4 angstroms).

    Returns:
    V (float): The potential energy:
               - V = ∞ if r < sigma (particles overlap),
               - V = 0 if r >= sigma (no overlap).
    """
    # set boundary conditions
    if r < sigma:
        V = np.inf
    else:
        V = 0
        
    return V

def square_well(r, sigma=3.4, epsilon=0.01, lam=1.5):
    """
    Calculate the intermolecular potential for a square-well interaction model.

    Parameters:
    r (float): The distance between two particles in angstroms.
    sigma (float, optional): The diameter of the particle (default is 3.4 angstroms).
    epsilon (float, optional): The depth of the potential well, representing the strength of attraction (default is 0.01 eV).
    lam (float, optional): The range of the attractive well, expressed as a multiple of sigma (default is 1.5).

    Returns:
    V (float): The potential energy:
               - V = ∞ if r < sigma (particles overlap),
               - V = -epsilon if sigma <= r < lambda * sigma (attractive region),
               - V = 0 if r >= lambda * sigma (no interaction).

    """
    # set boundary conditions
    if r < sigma:
        V = np.inf
    elif r >= (sigma * lam):
        V = 0
    else:
        V = -epsilon
        
    return V

def lennard_jones(r, epsilon=0.01, sigma=3.4):
    """
    Calculates the Lennard Jones potential using the Lennard Jones formula for particles
    at a distance r. Is set up for argon by default.

    Parameters:
    r (float): Distance between two atoms in angstroms.
    epsilon (float, optional): Depth of the potential well (i.e., the minimum energy).
    sigma (float, optional): Distance at which the potential is zero.

    Returns:
    float: Calculated Lennard Jones potential.
    """
    V = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    
    return V

def B2V(T, potential):
    """
    Calculate the second virial coefficient (B2V) for a given temperature and intermolecular potential.

    Parameters:
    T (float): The absolute temperature in Kelvin.
    potential (callable): A function representing the intermolecular potential. The potential function should take 
                          a distance (r) as input and return the corresponding potential energy. It must also have a 
                          'sigma' parameter defined by `inspect.signature(potential).parameters['sigma'].default`.

    Returns:
    B2V (float): The calculated second virial coefficient.
    """
    # define constants
    kB = 8.617e-5
    NA = 6.022e23
    
    # define space for integration
    sigma = inspect.signature(potential).parameters['sigma'].default
    space = np.linspace(0.001, 5 * sigma, 1000)
    
    # calculate values of function (to be integrated) at each x
    y = [(np.exp(-potential(x) / (kB * T)) - 1) * x ** 2 for x in space]
    
    B2V = -2 * np.pi * NA * scipy.integrate.trapezoid(y, space)
    
    return B2V

if __name__ == "__main__":

    # display header
    header = 'Second viral coefficients at 100 K for various potentials\n'
    header += '-' * (len(header) - 1)
    print(header)

    B2V_lj = B2V(100, lennard_jones)
    B2V_hs = B2V(100, hard_sphere)
    B2V_sw = B2V(100, square_well)

    # display results
    print(f'Hard-Sphere Potential: {B2V_hs} \nSquare-Well Potential: {B2V_sw} \nLennard-Jones Potential: {B2V_lj}')
