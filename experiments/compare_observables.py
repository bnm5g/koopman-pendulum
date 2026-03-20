import numpy as np
import matplotlib.pyplot as plt

from src.utils import compute_error, compute_rollout_error
from src.dynamics import pendulum_step, linearized_step
from src.observables import richer_observables as observables
from src.koopman import lift_data, compute_koopman

# =========================
# Data Generation
# =========================
def generate_data(n_trajectories=50, steps=200):
    X, Y = [], []

    for _ in range(n_trajectories):
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-2, 2)
        x = np.array([theta, omega])

        for _ in range(steps):
            x_next = pendulum_step(x)
            X.append(x)
            Y.append(x_next)
            x = x_next

    return np.array(X), np.array(Y)


# =========================
# Simulation Functions
# =========================
def simulate_true(x0, steps=200):
    traj = []
    x = x0.copy()

    for _ in range(steps):
        x = pendulum_step(x)
        traj.append(x)

    return np.array(traj)


def simulate_linear(x0, steps=200):
    traj = []
    x = x0.copy()

    for _ in range(steps):
        x = linearized_step(x)
        traj.append(x)

    return np.array(traj)


def predict_koopman(K, C, x0, steps=200):
    z = observables(x0)
    traj = []

    for _ in range(steps):
        z = K @ z
        x = C @ z
        traj.append(x)

    return np.array(traj)


# =========================
# Training Function
# =========================
def train_koopman(X_train, Y_train):
    Z = lift_data(X_train, observables)
    Z_next = lift_data(Y_train, observables)

    K = compute_koopman(Z, Z_next)

    # Reconstruction matrix
    C = X_train.T @ np.linalg.pinv(Z.T)

    return K, C


# =========================
# Main Experiment
# =========================
def run_experiment():
    # ---- Generate data ----
    X, Y = generate_data()

    # ---- Train/test split ----
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # ---- Train Koopman ----
    K, C = train_koopman(X_train, Y_train)
    print("Koopman matrix shape:", K.shape)

    # ---- Pick test initial condition ----
    x0 = X_test[0]

    # ---- Simulate ----
    true_traj = simulate_true(x0)
    koopman_traj = predict_koopman(K, C, x0)
    linear_traj = simulate_linear(x0)

    # ---- Compute errors ----
    error_mean = compute_error(true_traj, koopman_traj)
    error_curve = compute_rollout_error(true_traj, koopman_traj)

    print(f"Mean rollout error: {error_mean:.4f}")

    # ---- Plot phase space ----
    plt.figure()
    plt.plot(true_traj[:, 0], true_traj[:, 1], label="True")
    plt.plot(koopman_traj[:, 0], koopman_traj[:, 1], '--', label="Koopman")
    plt.plot(linear_traj[:, 0], linear_traj[:, 1], ':', label="Linear")

    plt.xlabel("theta")
    plt.ylabel("omega")
    plt.legend()
    plt.title("Phase Space: True vs Koopman")

    plt.savefig("results/plots/phase_comparison.png")
    plt.show()

    # ---- Plot error ----
    plt.figure()
    plt.plot(error_curve)
    plt.title("Rollout Error vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("Error")

    plt.savefig("results/plots/rollout_error.png")
    plt.show()


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    run_experiment()