import numpy as np

def pendulum_step(x, dt=0.01):
    theta, omega = x
    g, l = 9.81, 1.0

    return np.array([
        theta + dt * omega,
        omega - dt * (g/l) * np.sin(theta)
    ])