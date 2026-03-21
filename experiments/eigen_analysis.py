import numpy as np
import matplotlib.pyplot as plt

from src.koopman import compute_koopman_ridge
from src.observables import richer_observables as basic_observables
from src.data import generate_trajectories, split_trajectories, trajectories_to_dataset

np.random.seed(42)

trajectories = generate_trajectories()
train, test = split_trajectories(trajectories)
X, Y = trajectories_to_dataset(train)

Z = np.array([basic_observables(x) for x in X])
Z_next = np.array([basic_observables(y) for y in Y])

K = compute_koopman_ridge(Z, Z_next)

eigvals = np.linalg.eigvals(K)

plt.figure()
plt.scatter(eigvals.real, eigvals.imag)
plt.axhline(0)
plt.axvline(0)

plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Koopman Eigenvalues")

plt.savefig("results/plots/eigenvalues.png")
plt.show()