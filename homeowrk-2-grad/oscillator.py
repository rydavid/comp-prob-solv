import numpy as np
import matplotlib.pyplot as plt

# define constants in amu
h = 1                        # reduced plank constant in au
m = 1                        # mass of the oscillator in au
omega = 1                    # frequency of the oscillator in au
D = 10                       # depth of the potential well in au
beta = np.sqrt(1 / (2 * D))  # parameter controlling the width
                             # of the width of the potential well
L = 40                       # length of the potential well in au

# define grid from -L/2 to L/2
n = 2000  # number of points
x = np.linspace(-L / 2, L / 2, n)


def harmonic_potential(x, mass=m, omega=omega):
    """
    Compute the harmonic potential energy.

    Parameters:
    x (float): The position variable.
    mass (float, optional): The mass of the particle. Default is `m`.
    omega (float, optional): The angular frequency of the system. Default is `omega`.

    Returns:
    float: The harmonic potential energy at position `x`.
    """
    # compute harmonic potenial energy
    V = 0.5 * mass * (omega ** 2) * (x ** 2)
    
    return V


def anharmonic_potential(x, depth=D, beta=beta):
    """
    Compute the anharmonic potential energy using a Morse-like potential.

    Parameters:
    x (float): The position variable.
    depth (float, optional): The depth of the potential well. Default is `D`.
    beta (float, optional): The range parameter controlling the width of the potential well. Default is `beta`.

    Returns:
    float: The anharmonic potential energy at position `x`.
    """
    # compute anharmonic potenial energy
    V = depth * ((1 - np.exp(-beta * x)) ** 2)
    
    return V


def construct_potential_matrix(n, potential):
    """
    Construct a matrix with potential values along the diagonal.

    Parameters:
    n (array-like): A sequence of positions for which the potential values will be computed.
    potential (callable): A function that computes the potential energy for a given position.

    Returns:
    np.ndarray: A diagonal matrix where the diagonal elements are the potential energy values at the positions `n`.
    """
    # compute potential values
    potential_vals = [potential(x) for x in n]
    
    # store values along diagonal of an nxn matrix
    potential_matrix = np.diag(potential_vals)
    
    return potential_matrix


def compute_laplacian(n):
    """
    Constructs the Laplacian matrix for a system with n points in space.

    Parameters:
    n (int): Length of each dimension in calculacted Laplacian.

    Returns:
    numpy.ndarray: Calculated Laplacian matrix of dimensions n x n.
    """
    # create identity matrices
    ident = np.identity(n)
    ident_up = np.diag(np.ones(n - 1), 1)
    ident_down = np.diag(np.ones(n - 1), -1)
    ident_off = ident_up + ident_down
    
    # define dx
    dx = 40 / (n - 1)
    
    # compute matrix
    lp = 1/(dx ** 2) * (-2 * ident + ident_off)
    
    return lp


def construct_hamiltonian(laplacian, potential):
    """
    Construct the Hamiltonian matrix of the system.

    Parameters:
    laplacian (np.ndarray): The Laplacian matrix representing the kinetic energy operator.
    potential (np.ndarray): The potential energy matrix.

    Returns:
    np.ndarray: The computed Hamiltonian matrix.
    """    
    H = -0.5 * laplacian + potential
    
    return H


# calculate the hamiltonian and potential at each point
H_harmonic = construct_hamiltonian(compute_laplacian(n), construct_potential_matrix(x, harmonic_potential))
harmonic = [harmonic_potential(r) for r in x]

# get eigenvalues and eigenvectors and sort them
eig_values, eig_vectors = np.linalg.eig(H_harmonic)
inds = np.argsort(eig_values)
eig_values = eig_values[inds]
eig_vectors = eig_vectors[:, inds]

for i in range(10):
    # find the bounds at the energy level
    length = len(harmonic)
    closest_ind_lower = np.abs(eig_values[i] - harmonic[:round(length / 2)]).argmin()
    closest_ind_upper = np.abs(eig_values[i] - harmonic[round(length / 2):]).argmin()
    closest_ind_upper += round(length / 2)
    
    # shift wavefunction to proper energy level
    y = eig_vectors[:, i]
    
    # store upper and lower domain bounds
    upper = L / 2
    lower = -L / 2
    
    # weight and shift wavefunction
    y = 5 * -y + eig_values[i]
    
    # add items to plot
    plt.plot(x, y)
    plt.hlines(eig_values[i], x[closest_ind_lower], x[closest_ind_upper])
    
plt.plot(x, harmonic)
plt.ylim(0, 13)
plt.xlim(-7.5, 7.5)
plt.xlabel('Distance from lowest depth ($au$)')
plt.ylabel('Energy ($au$)')
plt.title('First Ten Wavefunctions for a Particle \n in a Harmonic Potential Well')
plt.savefig('harmonic_oscillator.tiff', format='tiff', dpi=300)
plt.show()

# calculate the hamiltonian and potential at each point
H_anharmonic = construct_hamiltonian(compute_laplacian(n), construct_potential_matrix(x, anharmonic_potential))
anharmonic = [anharmonic_potential(r) for r in x]

# get eigenvalues and eigenvectors and sort them
eig_values, eig_vectors = np.linalg.eig(H_anharmonic)
inds = np.argsort(eig_values)
eig_values = eig_values[inds]
eig_vectors = eig_vectors[:, inds]

for i in range(10):
    # find the bounds at the energy level
    length = len(anharmonic)
    closest_ind_lower = np.abs(eig_values[i] - anharmonic[:round(length / 2)]).argmin()
    closest_ind_upper = np.abs(eig_values[i] - anharmonic[round(length / 2):]).argmin()
    closest_ind_upper += round(length / 2)
    
    # weight wavefunction and shift to proper energy level
    y = eig_vectors[:, i]
    y = 3.5 * y + eig_values[i]
            
    # add items to plot
    plt.plot(x, y)
    plt.hlines(eig_values[i], x[closest_ind_lower], x[closest_ind_upper])

plt.plot(x, anharmonic)
plt.xlabel('Distance from lowest depth ($au$)')
plt.ylabel('Energy ($au$)')
plt.ylim(0, 11)
plt.xlim(-5, 20)
plt.title('First Ten Wavefunctions for a Particle \n in an Anharmonic Potential Well')
plt.savefig('anharmonic_oscillator.tiff', format='tiff', dpi=300)
plt.show()