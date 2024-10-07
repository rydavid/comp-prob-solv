import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import os

def linear_fn(params, x):
    a, b = params
    return a * x + b

def objective_fn(params, x, y):
    a, b = params
    y_model = linear_fn(params, x)
    ssr =  np.sum((y_model - y) ** 2)
    return ssr

if __name__ == "__main__":
    # read the csv into a pandas dataframe
    dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir, 'trouton.csv')
    df = pd.read_csv(path)

    # separate each class by creating arrays containing dictionaries for each substance in each class
    perfect_liquids = [df.loc[group].to_dict() for group in df[df['Class'] == 'Perfect liquids'].index]
    imperfect_liquids = [df.loc[group].to_dict() for group in df[df['Class'] == 'Imperfect liquids'].index]
    quantum_liquids = [df.loc[group].to_dict() for group in df[df['Class'] == 'Liquids subject to quantum effects'].index]
    metals = [df.loc[group].to_dict() for group in df[df['Class'] == 'Metals'].index]

    # store classes and associated colors and labels in same-size arrays
    liquids = [perfect_liquids, imperfect_liquids, quantum_liquids, metals]
    colors = ['Blue', 'Green', 'Red', 'Silver'] 
    labels = ['Perfect liquids', 'Imperfect liquids', 'Liquids subject to quantum effects', 'Metals']

    # minimize the sum of square residuals
    x = df['T_B (K)']
    y = df['H_v (kcal/mol)'] * 4184
    initial_guess = [1, 1]
    result = scipy.optimize.minimize(objective_fn, initial_guess, args=(x, y))
    slope, intercept = result.x 

    T_val = []
    H_val = []

    # plot the data
    plt.figure()
    plt.rcParams['text.usetex'] = True
    for liquid_list, color, label in zip(liquids, colors, labels):
        first = True
        for liquid in liquid_list:
            if first:
                plt.plot(liquid['T_B (K)'], liquid['H_v (kcal/mol)'] * 4184, marker='o', color=color, label=label)
                first = False
            else:
                plt.plot(liquid['T_B (K)'], liquid['H_v (kcal/mol)'] * 4184, marker='o', color=color)
            
            # Append values for regression
            T_val.append(liquid['T_B (K)'])
            H_val.append(liquid['H_v (kcal/mol)'] * 4184)

    y = np.array([slope * x + intercept for x in T_val])  # since there's linear interpolation between pts

    # add info to plot and display
    equation = fr'$H_v = {slope:.3f} T_B {intercept:.3f}$'
    plt.plot(T_val, y, label=equation, color='Black')
    plt.legend()
    plt.xlabel('$T_B (J/mol-K)$')
    plt.ylabel('$H_v$')
    plt.title("Trouton's Rule Optimization")
    plt.show()
    