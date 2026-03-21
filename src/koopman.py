import numpy as np

def lift_data(X, lift_fn):
    return np.array([lift_fn(x) for x in X])

def compute_koopman_ridge(Z, Z_next, reg=1e-6):
    """
    Compute Koopman operator using ridge regression (EDMD).

    Z:       (N, D)
    Z_next:  (N, D)

    Returns:
        K: (D, D)
    """
    D = Z.shape[1]

    A = Z.T @ Z + reg * np.eye(D)      # (D, D)
    B = Z.T @ Z_next                  # (D, D)

    K = np.linalg.solve(A, B)         # more stable than inverse

    return K