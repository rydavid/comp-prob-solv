import numpy as np

def psi_2p_z(x,y,z):
    """
    Computes the value of the hydrogen atom 2p orbital in Cartesian coordinates.

    Parameters:
        x (float or array-like): The x-coordinate(s) where the wavefunction is evaluated.
        y (float or array-like): The y-coordinate(s) where the wavefunction is evaluated.
        z (float or array-like): The z-coordinate(s) where the wavefunction is evaluated.

    Returns:
        psi (float or array-like): The value(s) of the 2p_z orbital wavefunction at the specified (x, y, z) coordinates.
    """
    # convert from cartesian to polar coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    cos_theta = z / r
    
    # compute the value of the wavefunction
    psi = 1 / (4 * np.sqrt(2 * np.pi)) * r * cos_theta * np.exp(-r / 2)
    
    return psi