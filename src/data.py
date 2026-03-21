import numpy as np
from src.dynamics import pendulum_step


def generate_trajectories(n_trajectories=50, steps=200):
    """
    Generate trajectories of pendulum dynamics.

    Returns:
        list of arrays, each shape (steps+1, 2)
    """
    trajectories = []

    for _ in range(n_trajectories):
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-2, 2)

        x = np.array([theta, omega])
        traj = [x.copy()]

        for _ in range(steps):
            x = pendulum_step(x)
            traj.append(x.copy())

        trajectories.append(np.array(traj))

    return trajectories


def split_trajectories(trajectories, train_ratio=0.8):
    """
    Split trajectories into train/test sets.
    """
    n_train = int(train_ratio * len(trajectories))

    train = trajectories[:n_train]
    test = trajectories[n_train:]

    return train, test


def trajectories_to_dataset(trajectories):
    """
    Convert trajectory list into (X, Y) pairs.

    Returns:
        X: (N, 2)
        Y: (N, 2)
    """
    X, Y = [], []

    for traj in trajectories:
        X.append(traj[:-1])
        Y.append(traj[1:])

    return np.vstack(X), np.vstack(Y)