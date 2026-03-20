import numpy as np

def compute_error(true, pred):
    return np.mean(np.linalg.norm(true - pred, axis=1))

def compute_rollout_error(true, pred):
    return np.linalg.norm(true - pred, axis=1)