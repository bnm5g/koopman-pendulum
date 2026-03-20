import numpy as np
import matplotlib.pyplot as plt

from src.dynamics import pendulum_step
from src.observables import richer_observables as basic_observables
from src.koopman import lift_data, compute_koopman

# -------- Generate data --------
def generate_data(n_trajectories=50, steps=200):
    X, Y = [], []

    for _ in range(n_trajectories):
        # random initial condition
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-2, 2)
        x = np.array([theta, omega])

        for _ in range(steps):
            x_next = pendulum_step(x)
            X.append(x)
            Y.append(x_next)
            x = x_next

    return np.array(X), np.array(Y)
# -------- Train Koopman --------
X, Y = generate_data()

Z = lift_data(X, basic_observables)
Z_next = lift_data(Y, basic_observables)

K = compute_koopman(Z, Z_next)

# Learn reconstruction: x ≈ C z
C = X.T @ np.linalg.pinv(Z.T)   # shape (2, dim_z)

print("Koopman matrix shape:", K.shape)

# -------- Prediction --------
def predict_koopman(K, C, x0, steps=200):
    z = basic_observables(x0)
    traj = []

    for _ in range(steps):
        z = K @ z
        x = C @ z   # proper reconstruction
        traj.append(x)

    return np.array(traj)

def simulate_true(x0, steps=200):
    traj = []
    x = x0.copy()

    for _ in range(steps):
        x = pendulum_step(x)
        traj.append(x)

    return np.array(traj)

# -------- Compare --------
x0 = np.array([1.0, 0.0])

true_traj = simulate_true(x0)
koopman_traj = predict_koopman(K, C, x0)

# -------- Plot --------
plt.figure()

plt.plot(true_traj[:,0], true_traj[:,1], label="True")
plt.plot(koopman_traj[:,0], koopman_traj[:,1], '--', label="Koopman")

plt.xlabel("theta")
plt.ylabel("omega")
plt.legend()
plt.title("Phase Space: True vs Koopman")

plt.savefig("results/plots/phase_comparison.png")
plt.show()