import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import os


def ols_slope(x, y):
    """
    Calculate the slope of the ordinary least squares (OLS) linear regression.

    Parameters:
    x (array-like): Independent variable data.
    y (array-like): Dependent variable data.

    Returns:
    float: Slope of the OLS regression line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator


def ols_intercept(x, y):
    """
    Calculate the intercept of the ordinary least squares (OLS) linear regression.

    Parameters:
    x (array-like): Independent variable data.
    y (array-like): Dependent variable data.

    Returns:
    float: Intercept of the OLS regression line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean


def ols(x, y):
    """
    Compute the slope and intercept of the OLS linear regression.

    Parameters:
    x (array-like): Independent variable data.
    y (array-like): Dependent variable data.

    Returns:
    tuple: A tuple containing the slope and intercept of the OLS regression line.
    """
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept


def ssr(residuals):
    """
    Calculate the sum of squared residuals (SSR) for a regression model.

    Parameters:
    residuals (array-like): Residuals (errors) from the regression model.

    Returns:
    float: Sum of squared residuals.
    """
    return np.sum(residuals ** 2)


def variance(residuals):
    """
    Calculate the variance of residuals for a regression model.

    Parameters:
    residuals (array-like): Residuals (errors) from the regression model.

    Returns:
    float: Variance of residuals.
    """
    return ssr(residuals) / (len(residuals) - 2)


def se_slope(x, residuals):
    """
    Calculate the standard error of the slope for a linear regression model.

    Parameters:
    x (array-like): Independent variable data.
    residuals (array-like): Residuals (errors) from the regression model.

    Returns:
    float: Standard error of the slope.
    """
    # numerator
    numerator = variance(residuals)
    # denominator
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)


def se_intercept(x, residuals):
    """
    Calculate the standard error of the intercept for a linear regression model.

    Parameters:
    x (array-like): Independent variable data.
    residuals (array-like): Residuals (errors) from the regression model.

    Returns:
    float: Standard error of the intercept.
    """
    # numerator
    numerator = variance(residuals)
    # denominator
    x_mean = np.mean(x)
    denominator = len(x) * np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)


def confidence_interval_slope(x, residuals, confidence_level):
    """
    Calculate the confidence interval for the slope of a linear regression model.

    Parameters:
    x (array-like): Independent variable data.
    residuals (array-like): Residuals (errors) from the regression model.
    confidence_level (float): Desired confidence level (e.g., 0.95 for 95% confidence).

    Returns:
    float: The margin of error for the slope at the given confidence level.
    """
    # Calculate the standard error of the slope
    se = se_slope(x, residuals)

    # Calculate the critical t-value
    n_data_points = len(x)
    df = n_data_points - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)

    # Calculate the confidence interval
    return critical_t_value * se


def confidence_interval_intercept(x, residuals, confidence_level):
    """
    Calculate the confidence interval for the intercept of a linear regression model.

    Parameters:
    x (array-like): Independent variable data.
    residuals (array-like): Residuals (errors) from the regression model.
    confidence_level (float): Desired confidence level (e.g., 0.95 for 95% confidence).

    Returns:
    float: The margin of error for the intercept at the given confidence level.
    """
    # Calculate the standard error of the intercept
    se = se_intercept(x, residuals)

    # Calculate the critical t-value
    n_data_points = len(x)
    df = n_data_points - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)

    # Calculate the confidence interval
    return critical_t_value * se

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

    # initialize arrays for H and T values
    H_val = []
    T_val = []

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

    # linearly interpret the data using ordinary least squares
    slope, intercept = ols(T_val, H_val)
    y = np.array([slope * x + intercept for x in T_val])  # since there's linear interpolation between pts

    # calculate 95% confidence intervals
    residuals = np.array([y[i] - H_val[i] for i in range(len(y))])
    ci_slope = confidence_interval_slope(T_val, residuals, 0.95)
    ci_intercept = confidence_interval_intercept(T_val, residuals, 0.95)

    # add info to plot and display
    equation = fr'$H_v = {slope:.3f}(\pm{ci_slope:.3f}) T_B {intercept:.3f}(\pm{ci_intercept:.3f})$'
    plt.plot(T_val, y, label=equation, color='Black')
    plt.legend()
    plt.xlabel('$T_B (J/mol-K)$')
    plt.ylabel('$H_v$')
    plt.title("Trouton's Rule")
    plt.show()