import numpy as np

def basic_observables(x):
    theta, omega = x
    return np.array([
        theta,
        omega,
        theta**2,
        omega**2,
        np.sin(theta),
        np.cos(theta)
    ])

def richer_observables(x):
    theta, omega = x
    return np.array([
        theta,
        omega,
        theta**2,
        omega**2,
        theta**3,
        omega**3,
        np.sin(theta),
        np.cos(theta),
        theta * omega,
        np.sin(2*theta),
        np.cos(2*theta)
    ])