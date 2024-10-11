from comp_ther_prop import lennard_jones_pf, calculate_C_v
import numpy as np
import matplotlib.pyplot as plt

# set up temperature array
T = np.linspace(1, 200, 100000)

# calculate the heat capacity at constant temperature for each temp in T
C_v = calculate_C_v(lennard_jones_pf, T)

# plot
plt.figure()
plt.plot(T, C_v)
plt.grid()
plt.xlabel("Temperature (K)")
plt.ylabel("$C_v$ (eV/K)")
plt.title("Heat Capacity at Constant Volume vs. Temperature")
plt.show()