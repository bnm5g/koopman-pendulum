import numpy as np
import matplotlib.pyplot as plt

from src.koopman import compute_koopman
from src.observables import richer_observables as basic_observables
from experiments.compare_observables import generate_data

X, Y = generate_data()

Z = np.array([basic_observables(x) for x in X])
Z_next = np.array([basic_observables(y) for y in Y])

K = compute_koopman(Z, Z_next)

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