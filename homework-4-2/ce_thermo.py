import scipy
import numpy as np

def Ce3_isolated_pf(T):
    """
    Calculate the partition function for an isolated Ce(III) ion.

    Parameters:
    T (float): Temperature in Kelvin (unused in this case but necessary to be used in thermodynamic_properties).

    Returns:
    int: The partition function value, which is 14.
    """
    return 14

def Ce3_SOC_pf(T):
    """
    Calculate the partition function for a Ce(III) ion considering spin-orbit coupling (SOC).

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    float: The partition function value for the Ce(III) ion with SOC.
    """
    k = scipy.constants.Boltzmann / scipy.constants.eV  # boltzmann constant in eV
    pf = 6 + 8 * np.exp(-0.28 / (k * T))
    return pf

def Ce3_SOC_CFS_pf(T):
    """
    Calculate the partition function for a Ce(III) ion considering both spin-orbit coupling (SOC) 
    and crystal field splitting (CFS).

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    float: The partition function value for the Ce(III) ion with SOC and CFS.
    """
    k = scipy.constants.Boltzmann / scipy.constants.eV  # boltzmann constant in eV
    pf = 4 + 2 * np.exp(-0.12 / (k * T)) + 2 * np.exp(-0.25 / (k * T)) + 4 * np.exp(-0.32 / (k * T)) + 2 * np.exp(-0.46 / (k * T))
    return pf

def thermodynamic_properties(Z_func, T_vals):
    """
    Calculate internal energy, free energy, and entropy of a thermodynamic system.

    Parameters:
    Z_func (function): A function that takes T and returns the partition function Z.
    T_vals (array-like): Temperatures in Kelvin.

    Returns:
    U (float): Internal energy.
    F (float): Helmholtz free energy.
    S (float): Entropy.
    """
    k = scipy.constants.Boltzmann / scipy.constants.eV  # boltzmann constant in eV

    # calculate partition function Z and its logarithm
    Z = np.array([Z_func(T) for T in T_vals])
    ln_Z = np.log(Z)

    # Calculate betas
    betas = 1 / (k * T_vals)

    # calculate thermodynamic properties
    U = -np.gradient(ln_Z, betas)
    F = -k * T_vals * ln_Z
    S = -np.gradient(F, T_vals)

    return U, F, S

# set temperature scale
T_vals = np.linspace(300, 2000, 5000)

# calculate thermodynamic properties
isolated = thermodynamic_properties(Ce3_isolated_pf, T_vals)
SOC = thermodynamic_properties(Ce3_SOC_pf, T_vals)
SOC_CFS = thermodynamic_properties(Ce3_SOC_CFS_pf, T_vals)

# save the property values as .csv files for each system
iso_columns = np.column_stack((T_vals, isolated[0], isolated[1], isolated[2]))
labels = 'Temperature (K), Internal Energy (eV), Free Energy (eV), Entropy (eV / K)'
np.savetxt('Ce(III)_isolated.csv', iso_columns, delimiter=',', header=labels, comments='', fmt='%.6e')
SOC_columns = np.column_stack((T_vals, SOC[0], SOC[1], SOC[2]))
np.savetxt('Ce(III)_SOC.csv', SOC_columns, delimiter=',', header=labels, comments='', fmt='%.6e')
SOC_CFS_columns = np.column_stack((T_vals, SOC_CFS[0], SOC_CFS[1], SOC_CFS[2]))
np.savetxt('Ce(III)_SOC_CFS.csv', SOC_CFS_columns, delimiter=',', header=labels, comments='', fmt='%.6e')
