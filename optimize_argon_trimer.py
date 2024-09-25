import numpy as np  
import scipy
import matplotlib.pyplot as plt
import os
from optimize_argon_dimer import lennard_jones, compute_bond_length, compute_bond_angles, save_xyz_file


def total_lennard_jones(r, epsilon=0.01, sigma=3.4):
    """
    Calculates the total Lennard Jones potential using the Lennard Jones formula.
    Is set up for argon by default.

    Parameters:
    r (list): Distances between pairs of atoms in angstroms.
    epsilon (float, optional): Depth of the potential well (i.e., the minimum energy).
    sigma (float, optional): Distance at which the potential is zero.

    Returns:
    float: Calculated Lennard Jones potential.
    """
    V = np.sum([lennard_jones(distance) for distance in r])
    return V


def fitting_fn(params):
    """
    Calculate the Lennard-Jones potential for a given set of parameters.

    Parameters:
    params (list or tuple):
        Containing the following values:
        - r12 (float): The distance between the first and second atoms.
        - x3 (float): The x-coordinate of the third atom.
        - y3 (float): The y-coordinate of the third atom.

    Returns:
    float: The calculated Lennard-Jones potential for the defined atomic configuration.
    
    Notes:
    This is a fitting function used in conjunction with scipy.optimize to determine params at minimal V.
    """
    r12, x3, y3 = params
    Ar0 = [0,   0]
    Ar1 = [r12, 0]
    Ar2 = [x3, y3]
    
    r23 = compute_bond_length([Ar1, Ar2])
    r13 = compute_bond_length([Ar0, Ar2])
    
    r = [r12, r23, r13]
    V = total_lennard_jones(r)
    return V


if __name__ == "__main__":
    # calculate the minimum lennard_jones potential for trimeric argon
    minimized_Ar3 = scipy.optimize.minimize(fitting_fn, [4, 2, 2])

    # define atomic coordinates
    Ar0 = [0, 0, 0]
    Ar1 = [minimized_Ar3.x[0], 0, 0]
    Ar2 = [minimized_Ar3.x[1], minimized_Ar3.x[2], 0]
    coordinates = [Ar0, Ar1, Ar2]
    
    # save xyz file
    atoms = 'Ar', 'Ar', 'Ar'
    filename = 'argon_trimer.xyz'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, filename)
    save_xyz_file(path, atoms, coordinates, name='trimeric argon')
    print(f'{filename} saved to {path}\n')

    # display header
    header = 'Trimeric argon properties calculated from Lennard Jones potential\n'
    header += '-' * (len(header) - 1)
    print(header)

    # calculate and display bond lengths
    r12 = compute_bond_length([Ar0, Ar1])
    r13 = compute_bond_length([Ar0, Ar2])
    r23 = compute_bond_length([Ar1, Ar2])
    print(f'Optimal bond lengths: \n* r12 = {r12:.3f} \n* r13 = {r13:.3f} \n* r23 = {r23:.3f}')

    # calculate and print bond angles for trimeric argon
    angles = compute_bond_angles(coordinates)
    print(f'Optimal bond angles: \n* bond angle 1 = {angles[0]:.2f} degrees \n* bond angle 2 = {angles[1]:.2f} degrees \n* bond angle 3 = {angles[0]:.2f} degrees')

    # The atoms are in what is very, very close to an arrangement resembling an equilateral triangle
    # with some minor deviation lost to imperfect numerical precision (and obviously fitting): 
    # print(f"bond angle 1 = {angles[0]} degrees \nbond angle 2 = {angles[1]} degrees \nbond angle 3 = {angles[0]} degrees")