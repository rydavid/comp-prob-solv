import numpy as np
from scipy.integrate import trapezoid

def isothermal_work(V_initial, V_final, n, T, pts=1000):
    """
    Computes the work performed during an isothermal expansion of an ideal gas
    using the trapezoidal integration method.
    
    Parameters:
        V_initial (float): Initial volume (m^3)
        V_final (float): Final volume (m^3)
        n (float): Number of moles of gas
        T (float): Temperature (K)
        pts (int): Number of points for integration, default is 1000
    
    Returns:
        float: The work performed (Joules)
    """
    R = 8.314  # ideal gas constant in J/(mol*K)
    
    # set volumes and convert to pressure for an ideal gas
    V_values = np.linspace(V_initial, V_final, pts)
    P_values = n * R * T / V_values
    
    # compute the work done
    work = - trapezoid(P_values, V_values)
    
    return work

# initialize arrays to store the work values
iso_work_vals = []

# define conditions for an ideal gas
V_i = 0.1
T = 300
gamma = 1.4
n = 1
pts = 1000

# set volume range
V_range = np.linspace(V_i, V_i * 3, pts)

# loop over each final volume and compute the work
for V_final in V_range:
    iso_work = isothermal_work(V_i, V_final, n, T, pts)
    iso_work_vals.append(iso_work)

# save the work vs final volume values as .csv
columns = np.column_stack((V_range, iso_work_vals))
np.savetxt('isothermal_work_vs_final_volume.csv', columns, delimiter=',', header='Final volume (m^3), Isothermal work (J)', comments='', fmt='%d')