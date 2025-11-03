"""
Script to plot related to plot concept drift detection
Created by Cláudio Modesto
LASSE
"""

import os
import pathlib
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from river import drift

ROOT_DIR = "../delay_database" # path to root database directory
# Create the output directory
OUTPUT_PATH_NAME = "figures"
if not os.path.isdir(OUTPUT_PATH_NAME):
    pathlib.Path(OUTPUT_PATH_NAME).mkdir(parents=True, exist_ok=True)

window_sizes = [6800, 7000, 6800]
topologies = ["5G-Crosshaul", "Germany", "PASSION"]
sub_paths = ["5g_crosshaul", "germany", "passion"]
traffic_labels = ["Exponential", "Poisson", "Uniform", "Normal", "Congested"]
N_OF_TOPOLOGIES = 3
N_OF_PATTERN = 4
for i in range(1, N_OF_TOPOLOGIES+1):
    ds_path = f"{ROOT_DIR}/{sub_paths[i-1]}/experiment"
    all_flow_traffic = []
    plt.subplot(3, 1, i)
    print(f"{topologies[i-1]}")
    plt.title(f"{topologies[i-1]}")
    if i == 2:
        plt.ylabel("Goodput (Mbits/s)", fontsize=15)
    start = 0
    ds = tf.data.Dataset.load(f"{ds_path}_{i}00_cv/testing", compression="GZIP")
    for j in range(1, N_OF_PATTERN+2):
        if j==3:
            continue
        flow_traffic = np.array([])
        for ii, (features, label) in enumerate(iter(ds)):
            flow_traffic = np.append(flow_traffic, features["flow_traffic"].numpy())
        all_flow_traffic.extend(flow_traffic[start:]/1e3)
        plt.plot(np.arange(start, len(flow_traffic)), flow_traffic[start:]/1e3, label=traffic_labels[j-1])
        start = len(flow_traffic)-1
        #print(f"Concept drift ocurr at: {start}")
        if j < N_OF_PATTERN+1:
            new_ds = tf.data.Dataset.load(f"{ds_path}_{i}0{j}_cv/testing", compression="GZIP")
            ds = ds.concatenate(new_ds)

    kswin = drift.KSWIN(alpha=0.001, window_size=window_sizes[i-1], stat_size=1200, seed=42)

    delay = np.array([])
    drift_detected = []
    for idx, sample in enumerate(all_flow_traffic):
        kswin.update(sample)
        if kswin.drift_detected:
            plt.scatter(idx, sample, zorder=2, s=60, marker="X", color="black", label="Drift detected")
plt.xlabel("Time (s)", fontsize=15)
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1, -0.5), ncol=5)
plt.savefig("figures/flow_traffic.pdf", bbox_inches="tight")
plt.close()
