import matplotlib.pyplot as plt
import numpy as np
from compute_work_adiabatic import adiabatic_work
from compute_work_isothermal import isothermal_work

def plot_work(V_initial, V_max, n, T, gamma, pts=1000):
    """
    Plots the work performed during an isothermal and adiabatic 
    expansion of an ideal gas as a function of the final volume.
    
    Parameters:
        V_initial (float): Initial volume (m^3)
        V_max (float): Maximum final volume (m^3)
        n (float): Number of moles of gas
        T (float): Temperature (K)
        gamma (float): Adiabatic index
        pts (int): Number of points for integration, default is 1000
    """
    # initialize arrays to store the work values
    iso_work_vals = []
    adi_work_vals = []

    # set volume range
    V_range = np.linspace(V_initial, V_max, pts)

    # loop over each final volume and compute the work
    for V_final in V_range:
        iso_work = isothermal_work(V_initial, V_final, n, T, pts)
        adi_work = adiabatic_work(V_initial, V_final, n, T, gamma, pts)
        iso_work_vals.append(iso_work)
        adi_work_vals.append(adi_work)
    
    # plot the work as a function of the final volume
    plt.figure(figsize=(8, 6))
    plt.plot(V_range, iso_work_vals, label="Isothermal", color='Blue')
    plt.plot(V_range, adi_work_vals, label="Adiabatic", color='Red')
    plt.xlabel('Final Volume ($m^3$)')
    plt.ylabel('Work Done ($J$)')
    plt.title(f'Work vs Final Volume ($T = {T} K, n = {n} mol$)')
    plt.grid(True)
    plt.legend()
    plt.show()


# define conditions for an ideal gas
V_i = 0.1
T = 300
gamma = 1.4
n = 1

# plot
plot_work(V_i, 3 * V_i, n, T, gamma)
