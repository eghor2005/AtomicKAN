import numpy as np
from matplotlib import pyplot as plt


def Michalewicz_Function(X1, X2): # Michalewicz Function
    m = 10
    term1 = -np.sin(X1) * (np.sin(X1**2/np.pi) ** (2 * m))
    term2 = -np.sin(X2) * (np.sin(2 * X2**2/np.pi) ** (2 * m))
    return term1 + term2

def Eggholder_function(X1, X2): # Eggholder function
    term1 = -(X2 + 47)*np.sin(np.sqrt(np.abs(X2 + X1/2+47)))
    term2 = -X1*np.sin(np.sqrt(np.abs(X1 - (X2+47))))
    return term1 + term2

def Styblinski_Tang_function(X1, X2): # Styblinski-Tang function
    return 0.5 * (X1**4 - 16*X1**2 + 5*X1 + X2**4 - 16*X2**2 + 5*X2)

def Dixon_Price_Function(X1, X2): # Dixon-Price Function
    term1 = (X1 - 1)**2
    term2 = 2 * (2 * X2**2 -X1)**2
    return  term1 + term2

def Sphere_function(X1, X2):
    return X1**2  + X2**2

def Three_Hump_Camel_Function(X1, X2):
    return 2*X1**2 - 1.05*X1**4 + (X1**6)/6 + X1*X2 + X2**2

def Rosenbrock_function(X1, X2):
    return 100*(X2-X1**2)**2 + (X1 - 1)**2

def Easom_function(X1, X2):
    return -np.cos(X1)*np.cos(X2)*np.exp(-(X1 - np.pi)**2-(X2-np.pi)**2)

def Booth_function(X1, X2):
    return (X1 + 2 * X2 - 7) ** 2 + (2*X1 + X2 - 5)**2

def Bohachevsky_function(X1, X2):
    return X1**2 + 2*X2**2 - 0.3 * np.cos(2 * np.pi * X1) - 0.4 * np.cos(4 * np.pi * X2) + 0.7
