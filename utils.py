import matplotlib.pyplot as plt
import autograd.numpy as np
import pandas as pd
import os

plt.rcParams.update({
    "font.family": "serif",             # or "sans-serif", "monospace"
    "font.serif": ["Times New Roman"],  # or "Palatino", "Georgia", etc.
    "font.size": 12,                    # default text size
    "axes.titlesize": 13,               # title font size
    "axes.labelsize": 12,               # x/y label font size
    "legend.fontsize": 11,              # legend font size
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

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

# ---------- plotters for data-driven simulation experiments ----------
def plot_nsr_cpe_comparison(file_path, exclude: tuple[str, ...] = ("No learning",)):
    df = pd.read_csv(file_path)
    df = df.sort_values("NSR")
    x = df["NSR"].to_numpy(dtype=float)

    def is_triplet_head(col):
        return (
            col != "NSR"
            and f"{col} q30" in df.columns
            and f"{col} q70" in df.columns
        )

    color_map = {
        "No learning": "#6B7280",   # gray
        "N4SID": "#F59E0B",         # amber
        "PAST": "#0EA5E9",          # cyan
        "Gr(9)": "#14B8A6",         # teal
        "Gr(10)": "#8B5CF6",        # purple
        "Gr(11)": "#F97316",        # orange
        "Flag(8,...,15)": "#22C55E", # green
        "Flag(9, 10)": "#ef2d5a",   # red
    }
    all_models = [m for m in color_map.keys() if m in df.columns and m not in exclude]
    colors = {m: color_map[m] for m in all_models}

    plt.figure(figsize=(7, 5.5), dpi=300)
    for model in all_models:
        med = df[model].to_numpy(dtype=float)
        q30 = df[f"{model} q30"].to_numpy(dtype=float)
        q70 = df[f"{model} q70"].to_numpy(dtype=float)
        plt.plot(x, med, marker="o", linewidth=2, label=model, color=colors[model])
        plt.fill_between(x, q30, q70, color=colors[model], alpha=0.18)

    plt.xlabel("Noise-to-signal ratio (NSR)")
    plt.ylabel("Median cumulative prediction error")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=9)
    plt.title("CPE vs. NSR for Various Models")
    plt.tight_layout()
    plt.show()

def plot_trajectories(trajectory_1, trajectory_2, title):
    plt.figure(figsize=(6, 4))
    labels=("True", "Prediction")
    plt.plot(trajectory_1.ravel(), label=labels[0])
    plt.plot(trajectory_2.ravel(), label=labels[1])
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_median_prediction_errors(pred_error_trials, labels=None, title="Median Prediction Error Across Time"):
    num_trials, num_models, _ = pred_error_trials.shape
    x = np.arange(T_SIM)
    med_errors = np.median(pred_error_trials, axis=0)
    q30 = np.percentile(pred_error_trials, 30, axis=0)
    q70 = np.percentile(pred_error_trials, 70, axis=0)
    
    if labels is None:
        labels = [f"Model {i+1}" for i in range(num_models)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))
    plt.figure(figsize=(7, 4))
    for i in range(num_models):
        plt.plot(
            x, med_errors[i],
            label=labels[i],
            color=colors[i],
            linewidth=1.8
        )
        plt.fill_between(
            x, q30[i], q70[i],
            color=colors[i], alpha=0.25
        )
    plt.title(title, fontsize=12)
    plt.xlabel("Time step $t$", fontsize=11)
    plt.ylabel("Median Prediction Error", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_flag_vs_gr(file_path):
        df = pd.read_csv(file_path)
        dims = np.arange(8, 8 + len(df))  # 8–15

        # Extract columns
        fl_med = df.iloc[:, 1].to_numpy()
        fl_iqr = df.iloc[:, 2].to_numpy()
        gr_med = df.iloc[:, 3].to_numpy()
        gr_iqr = df.iloc[:, 4].to_numpy()
        x = np.arange(len(dims))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5.5))
        bars_fl = ax.bar(
            x - width/2, fl_med, width,
            yerr=fl_iqr/2, capsize=4,
            label="Flag", alpha=0.9, color="#3B82F6"
        )
        bars_gr = ax.bar(
            x + width/2, gr_med, width,
            yerr=gr_iqr/2, capsize=4,
            label="Gr", alpha=0.9, color="#10B981"
        )
    
        # Annotate numbers above bars
        for bar in bars_fl:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3), 
                        textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
        for bar in bars_gr:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("Subspace Dimension")
        ax.set_ylabel("Median Cumulative Error")
        ax.set_title("Flag vs Grassmann Cumulative Error (Median + IQR)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    
        plt.tight_layout()
        plt.show()

def plot_median_cum_errors_all_models(file_path):
        df = pd.read_csv(file_path)
        model_names = [
        "No learning", "N4SID", "PAST", "Flag(9,10)",
        "Gr(9)", "Gr(10)", "Gr(11)", "Flag(8,…,15)"
        ]  
        med = df.iloc[:, 0].to_numpy()
        q30 = df.iloc[:, 1].to_numpy()
        q70 = df.iloc[:, 2].to_numpy()
        yerr_lower = med - q30
        yerr_upper = q70 - med
        yerr = np.vstack([yerr_lower, yerr_upper])
        x = np.arange(len(model_names))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(x, med, color="#3B82F6", alpha=0.8, capsize=4)
        ax.errorbar(x, med, yerr=yerr, fmt='none', ecolor='black', capsize=4, elinewidth=1)

        # Add numeric labels on top of bars
        for rect, val in zip(bars, med):
            ax.text(
                rect.get_x() + rect.get_width()/2,
                rect.get_height() + 0.05,
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=9
            )

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha='right')
        ax.set_ylabel("Median Cumulative Error")
        ax.set_title("Median Cumulative Errors with IQR Whiskers")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()




