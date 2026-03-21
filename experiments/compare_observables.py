import numpy as np
import matplotlib.pyplot as plt

from src.utils import compute_error, compute_rollout_error
from src.dynamics import pendulum_step, linearized_step
from src.observables import richer_observables as observables
from src.koopman import lift_data, compute_koopman_ridge        
from src.data import generate_trajectories, split_trajectories, trajectories_to_dataset

np.random.seed(42)

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

    K = compute_koopman_ridge(Z, Z_next)

    C = X_train.T @ np.linalg.pinv(Z.T)

    return K.T, C


# =========================
# Main Experiment
# =========================
def run_experiment():
    # ---- Generate data ----
    trajectories = generate_trajectories()
    np.random.shuffle(trajectories)
    train_trajs, test_trajs = split_trajectories(trajectories)
    X_train, Y_train = trajectories_to_dataset(train_trajs)
    X_test, Y_test = trajectories_to_dataset(test_trajs)

    # ---- Train Koopman ----
    K, C = train_koopman(X_train, Y_train)
    print("Koopman matrix shape:", K.shape)

    # ---- Pick test initial condition ----
    errors = []
    linear_errors = []

    for traj in test_trajs[:20]:
        x0 = traj[0]

        true_traj = simulate_true(x0)
        koopman_traj = predict_koopman(K, C, x0)
        linear_traj = simulate_linear(x0)

        errors.append(compute_error(true_traj, koopman_traj))
        linear_errors.append(compute_error(true_traj, linear_traj))

    print("Koopman avg error:", np.mean(errors))
    print("Linear avg error:", np.mean(linear_errors))

    # ---- Simulate ----
    x0 = test_trajs[0][0]
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