import numpy as np

def pendulum_step(x, dt=0.01):
    def f(x):
        theta, omega = x
        g, l = 9.81, 1.0
        return np.array([omega, -(g/l)*np.sin(theta)])

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def linearized_step(x, dt=0.01):
    theta, omega = x

    # small-angle approximation: sin(theta) ≈ theta
    return np.array([
        theta + dt * omega,
        omega - dt * theta
    ])