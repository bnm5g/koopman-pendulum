import numpy as np

def lift_data(X, lift_fn):
    return np.array([lift_fn(x) for x in X])

def compute_koopman(Z, Z_next):
    return Z_next.T @ np.linalg.pinv(Z.T)