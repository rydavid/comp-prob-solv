import numpy as np  
import scipy
import matplotlib.pyplot as plt
import os


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


def compute_bond_length(coords):
    """
    Uses the distance formula to calculate the bond length between two covalently bound atoms.

    Parameters:
    coord (list): Cartesian coordinates of two bound atoms.

    Returns:
    float: Calculated bond length in angstroms.
    """
    # expand coords
    coord1 = coords[0]
    coord2 = coords[1]
    
    # calculate bond length
    bond_length = np.sqrt(sum([(coord1[i] - coord2[i]) ** 2 for i in range(len(coord1))]))
    
    return bond_length


def compute_bond_angles(coords):
    """
    Calculates the bond angle between all atoms in a molecule (assuming they are all bound to one another).

    Parameters:
    coords (list): 2D list of cartesian coordinates corresponding to the locations of atoms.
    
    Returns:
    list: Calculated bond angles in degrees.
    """
    # initialize thetas list and triplets dictionary
    thetas = []
    triplets = {}
    
    # return 180 degrees for diatomics
    if len(coords) == 2:
        return 180
    
    # else loop through the coordinates and compute bond angles
    else:
        for coord1 in coords:
            for coord2 in coords:
                if coord1 != coord2:
                    for coord3 in coords:
                        if coord1 != coord3 and coord2 != coord3:
                            # convert lists to NumPy arrays
                            a = np.array(coord1)
                            central_atom = np.array(coord2)
                            b = np.array(coord3)

                            # calculate bond angle
                            theta = np.degrees(np.arccos(np.dot(a - central_atom, b - central_atom) / (np.linalg.norm(a - central_atom) * np.linalg.norm(b - central_atom))))

                            # add theta to thetas if bond angle isn't already in it
                            key = (tuple(coord1), tuple(coord2), tuple(coord3))
                            backward_key = key[::-1]

                            # check if angle is already in list, and if not, add it
                            if not key in triplets and not backward_key in triplets:
                                triplets[key] = theta
                                thetas.append(theta)
    
        return thetas


def save_xyz_file(filename, atoms, coordinates, name=None):
    """
    Save the atomic structure in XYZ format.
    
    Parameters:
    filename (str): The name of the output file.
    atoms (list of str): List of atom types.
    coordinates (list of list): List of atomic coordinates.
    """
    num_atoms = len(atoms)
    
    # open the file for writing
    with open(filename, 'w') as file:
        # write number of atoms as the first line
        file.write(f"{num_atoms}\n")
        
        # make comment line optional
        if name != None:
            file.write(f"{name}\n")
        else:
            file.write("\n")
        
        # write atom types and coordinates
        for atom, coord in zip(atoms, coordinates):
            file.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

            
if __name__ == "__main__":
    # calculate the minimum lennard_jones potential
    minimized = scipy.optimize.minimize(lennard_jones, 4)
    
    # set atomic coordinates
    Ar0 = [0, 0, 0]
    Ar1 = [minimized.x[0], 0, 0]
    coordinates = Ar0, Ar1
    
    # calculate bond length and bond angle
    bond_length = compute_bond_length(coordinates)
    bond_angle = compute_bond_angles(coordinates)
    
    # save xyz file and display
    atoms = 'Ar', 'Ar'
    filename = 'argon_dimer.xyz'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, filename)
    save_xyz_file(path, atoms, coordinates, name='dimeric argon')
    print(f'{filename} saved to {path}\n')
    header = 'Dimeric argon properties calculated from Lennard Jones potential\n'
    header += '-' * (len(header) - 1)
    print(header)
    print(f'Optimal bond length: \n* bond length = {bond_length:.3f} angstroms')
    print(f'Optimal bond angle: \n* bond angle = {bond_angle:.3f} degrees')