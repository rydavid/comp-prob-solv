import numpy as np

def psi_1s(x,y,z, Z=1, a0=1):
    """
    Computes the value of the hydrogen atom 1s orbital wavefunction in Cartesian coordinates.

    Parameters:
        x (float or array-like): The x-coordinate(s) where the wavefunction is evaluated.
        y (float or array-like): The y-coordinate(s) where the wavefunction is evaluated.
        z (float or array-like): The z-coordinate(s) where the wavefunction is evaluated.
        Z (float, optional): The atomic number of the nucleus. Default is 1.
        a0 (float, optional): The Bohr radius. Default is 1 (atomic units).

    Returns:
        psi (float or array-like): The value(s) of the 1s orbital wavefunction at the specified (x, y, z) coordinates.
    """
    # calculate distance from center
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # compute the value of the wavefuntion
    psi = 1 / np.sqrt(np.pi * a0**3) * np.exp(-r / a0)
    
    return psi


def laplacian_psi_1s(x, y, z, Z=1, a0=1):
    """
    Computes the Laplacian of the hydrogen atom 1s orbital wavefunction in Cartesian coordinates.

    Parameters:
        x (float or array-like): The x-coordinate(s) where the Laplacian is evaluated.
        y (float or array-like): The y-coordinate(s) where the Laplacian is evaluated.
        z (float or array-like): The z-coordinate(s) where the Laplacian is evaluated.
        Z (float, optional): The atomic number of the nucleus. Default is 1.
        a0 (float, optional): The Bohr radius. Default is 1 (atomic units).

    Returns:
        laplacian (float or array-like): The value(s) of the Laplacian of the 1s orbital wavefunction at the specified (x, y, z) coordinates.
    """
    # calculate distance from center
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # compute laplacian
    laplacian = 1 / (a0**2) * psi_1s(x, y, z) - 2 / (r * a0) * psi_1s(x, y, z)
    
    return laplacian