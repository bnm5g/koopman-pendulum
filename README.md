# 🧠 Linearizing Nonlinear Dynamics with Koopman Operators

This project explores how **nonlinear dynamical systems** can be approximated using **linear operators in higher-dimensional spaces** via the **Koopman operator framework**.

We study the classical **nonlinear pendulum** and investigate how well Koopman-based models can capture its dynamics compared to the true system.

---

## 🚀 Project Overview

A nonlinear dynamical system:

[
x_{t+1} = F(x_t)
]

can be lifted into a higher-dimensional observable space:

[
z = \Phi(x)
]

where the dynamics become approximately linear:

[
z_{t+1} = K z_t
]

This project implements this idea using **Extended Dynamic Mode Decomposition (EDMD)**.

---

## ⚙️ Features

* Nonlinear pendulum simulation
* Koopman operator approximation via EDMD
* Custom observable (feature) design
* Phase space trajectory comparison
* Prediction error analysis
* Eigenvalue analysis of the Koopman operator

---

## 📁 Repository Structure

```
koopman-pendulum/
│
├── src/                # Core implementation
├── experiments/        # Experiment scripts
├── notebooks/          # Exploratory analysis
├── data/               # Generated data (optional)
├── results/
│   ├── plots/          # Figures
│   └── metrics/        # Quantitative results
├── README.md
└── requirements.txt
```

---

## 📊 Results

### Phase Space Comparison

The Koopman model approximates the nonlinear dynamics but shows:

* Distortion in trajectory shape
* Phase misalignment
* Drift over time

This indicates that the chosen observable basis is insufficient to fully linearize the system.

---

### Key Insight

The Koopman approximation depends heavily on the choice of observables.
Finite-dimensional approximations may fail to preserve stability and long-term behavior.

Increasing observable dimension alone does not improve Koopman predictions unless a proper reconstruction mapping from lifted space to state space is learned.

Using a single trajectory results in a rank-deficient approximation of the Koopman operator, preventing richer observable sets from improving predictive performance
---

## 🧪 How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run the main experiment:

```
python -m experiments.compare_observables
```

---

## 📈 Experiments

* **compare_observables.py**
  Compares true vs Koopman trajectories in phase space

* **eigen_analysis.py**
  Computes and visualizes Koopman eigenvalues

* **prediction_error.py** *(optional)*
  Tracks error growth over time

---

## 🔬 Discussion

Initial experiments show that:

* Basic polynomial and trigonometric observables are insufficient for capturing nonlinear pendulum dynamics
* The Koopman operator may exhibit eigenvalues outside the unit circle, leading to instability
* Prediction error grows over time due to model mismatch

---

## 🚧 Future Work

* Design richer observable spaces
* Implement neural Koopman models (autoencoder-based)
* Extend to chaotic systems (e.g., double pendulum)
* Apply control methods in Koopman space


