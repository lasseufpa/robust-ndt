"""
Script to plot related to plot concept drift detection
Created by Cláudio Modesto
LASSE
"""

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from river import drift

OUTPUT_PATH_NAME = "figures"
if not os.path.isdir(OUTPUT_PATH_NAME):
    pathlib.Path(OUTPUT_PATH_NAME).mkdir(parents=True, exist_ok=True)

window_sizes = [6800, 7000, 6800]
sub_paths = ["5g_crosshaul", "germany", "passion"]
topologies = ["5G-Crosshaul", "Germany", "PASSION"]
traffic_labels = ["Exponential", "Poisson", "Uniform", "Normal", "Congested"]
N_OF_TOPOLOGIES = 3
N_OF_PATTERN = 4
ROOT_DIR = "../delay_database"
for i in range(1, N_OF_TOPOLOGIES+1):
    all_flow_traffic = []
    start = 0
    ds_path = f"{ROOT_DIR}/{sub_paths[i-1]}/experiment"
    ds = tf.data.Dataset.load(f"{ds_path}_{i}00_cv/testing", compression="GZIP")
    for j in range(1, N_OF_PATTERN+2):
        if j==3:
            continue
        flow_traffic = np.array([])
        for ii, (features, label) in enumerate(iter(ds)):
            flow_traffic = np.append(flow_traffic, features["flow_traffic"].numpy())
        all_flow_traffic.extend(flow_traffic[start:]/1e3)
        start = len(flow_traffic)-1
        if j < N_OF_PATTERN+1:
            new_ds = tf.data.Dataset.load(f"{ds_path}_{i}0{j}_cv/testing", compression="GZIP")
            ds = ds.concatenate(new_ds)

    number_of_drifts = []
    for window_size in range(100, 7000, 100):
        kswin = drift.KSWIN(alpha=0.001, window_size=window_size, stat_size=int(window_size/4), seed=42)
        idx_of_drifts = []
        for idx, sample in enumerate(all_flow_traffic):
            kswin.update(sample)
            if kswin.drift_detected:
                idx_of_drifts.append(idx)
        number_of_drifts.append(len(idx_of_drifts))
    plt.plot(np.arange(100, 7000, 100), number_of_drifts, linewidth=3, label=topologies[i-1])
plt.xlabel("Window size", fontsize=15)
plt.ylabel("# Concept drift detected", fontsize=15)
plt.legend()
plt.grid()
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.savefig("figures/window_plot.pdf", bbox_inches="tight")
plt.close()
