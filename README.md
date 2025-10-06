# FRONT Experiments

This repository contains geodesic tracking and data-driven prediction experiments for the **FRONT (Flag-based Recursive Online Nested Tracking)** algorithm.  
The experiments demonstrate how subspace learning on the **flag manifold** can be used for system identification and data-driven prediction in time-varying systems.

---

## 🧩 Project Structure
FRONT_experiments/

├── flag.py                     # Implementation of the Flag manifold and FRONT algorithm

├── utils.py                    # Helper functions for data handling, plotting, etc.

├── data_driven_simulation.ipynb # Notebook for data-driven prediction experiments

├── geodesic_tracking.ipynb     # Notebook for subspace and geodesic tracking demonstrations

├── requirements.txt            # Dependencies for running the experiments

## ⚙️ Setup

Clone this repository and install dependencies:
```bash
git clone https://github.com/DianJin-Frederick/FRONT-experiments.git
cd FRONT-experiments
pip install -r requirements.txt
```

## Running Experiments
### Tip:
We recommend running these notebooks on [Google Colab](https://colab.research.google.com/) to avoid potential package or environment compatibility issues.
All dependencies listed in requirements.txt can be easily installed in Colab.

Run the main experiments on data-driven prediction using:
```bash
jupyter notebook data_driven_simulation.ipynb
```
Visualize the evolution of subspaces and the tracking behavior on the flag manifold:
```bash
jupyter notebook geodesic_tracking.ipynb
```
