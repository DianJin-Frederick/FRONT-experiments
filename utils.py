import matplotlib.pyplot as plt
import autograd.numpy as np
import os
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use("Qt5Agg")
import os
SAVE_FOLDER = 'videos'
os.makedirs(SAVE_FOLDER, exist_ok=True)

soft_colors = {
    "soft_red": "#E57373",   # Muted red
    "soft_blue": "#64B5F6",  # Muted blue
    "soft_green": "#81C784", # Muted green
    "soft_purple": "#9575CD", # Muted purple
    "soft_orange": "#FFB74D" # Muted orange
}

def save_data(data, filename, save_folder):
    np.savetxt(
        os.path.join(save_folder, f"{filename}.csv"),
        data,  # your true outputs
        delimiter=","
    )

def projector(U):
    return U @ U.T

def chordal_distance(A, B):
    nonzero_singular_values = np.linalg.svd(A.T @ B, full_matrices=False)[1]
    distance = np.linalg.norm(np.sin(np.arccos(np.clip(nonzero_singular_values, -1, 1))))

    return np.sqrt(distance)

def chordal_metric(A, B):
    """
    Ye, Ke, and Lek-Heng Lim.
    "Schubert varieties and distances between subspaces of different dimensions."
    SIAM Journal on Matrix Analysis and Applications 37.3 (2016): 1176-1197.
    Equation 18
    """
    nonzero_singular_values = np.linalg.svd(A.T @ B, full_matrices=False)[1]
    distance = np.linalg.norm(np.sin(np.arccos(np.clip(nonzero_singular_values, -1, 1))))
    dim_gap = np.abs(np.linalg.matrix_rank(A) - np.linalg.matrix_rank(B))

    return dim_gap + np.sqrt(distance)

def compute_vector_angles(A, B):
    angles = []
    for i in range(min(A.shape[1], B.shape[1])):
        u = A[:, i]
        v = B[:, i]
        cosine = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        if angle >= 90:
            angle = 180 - angle
        angles.append(angle)

    return np.array(angles)

def compute_principal_angles(A, B):
    M = A.T @ B
    sigma = np.linalg.svd(M, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    principal_angles = np.degrees(np.arccos(sigma))
    for j in range(len(principal_angles)):
        if principal_angles[j] >= 90:
            principal_angles[j] = 180 - principal_angles[j]

    return principal_angles







