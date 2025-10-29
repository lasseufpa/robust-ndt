"""
Script to create paper single plot (NMSE)
Created by Cláudio Modesto
"""

import os
import pathlib
import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = "../results" # path to root database directory
# Create the output directory
OUTPUT_PATH_NAME = "figures"
if not os.path.isdir(OUTPUT_PATH_NAME):
    pathlib.Path(OUTPUT_PATH_NAME).mkdir(parents=True, exist_ok=True)

all_cd = [[68, 131, 194], [72, 143, 211], [72, 135, 198]]
nmse_w_drift = np.array([])

# load data from all npz file
for i in range(10):
    with open(f"../results/5g_crosshaul/results_sync_jitter_True_r_{i}.npz", "rb") as f:
        if nmse_w_drift.size == 0:
            nmse_w_drift = np.load(f)["arr_0"]
        else:
            nmse_w_drift = np.vstack([nmse_w_drift, np.load(f)["arr_0"]])
        if i == 0:
            drift_detected = np.load(f)["arr_1"]
            model_updated = np.load(f)["arr_2"]

with open("../results/5g_crosshaul/results_sync_jitter_False_r_0.npz", "rb") as f:
    nmse_wo_drift = np.load(f)["arr_0"]

avg_nmse_drift = np.mean(nmse_w_drift, axis=0)
std_nmse_drift = np.std(nmse_w_drift, axis=0)


plt.figure(figsize=(7, 5))
plt.plot(avg_nmse_drift, label="VTwin w/ retraining", linewidth=2)
plt.plot(nmse_wo_drift, label="VTwin w/o retraining", color="k", alpha=0.6, linestyle="--")
plt.fill_between(np.arange(avg_nmse_drift.size), avg_nmse_drift - std_nmse_drift,
                        avg_nmse_drift + std_nmse_drift,
                        color='blue', alpha=0.25, label='Standard deviation')

# adding index when the drift was detected
for index in drift_detected[:-1]:
    plt.axvline(x=index-1, color="r", linestyle="dotted")
plt.axvline(x=drift_detected[-1]-1, color="r", linestyle="dotted", label="Concept drift detected")

for index in model_updated[:-1]:
    plt.scatter(index, avg_nmse_drift[index], marker='x', color='red', zorder=2, s=79)
plt.scatter(model_updated[-1], avg_nmse_drift[-1], marker='x', color='red', zorder=2, s=79, label='Model updated')


for i in range(len(all_cd[0]) - 1):
    plt.axvline(x=all_cd[0][i], color="g", linestyle="dotted")
plt.axvline(x=all_cd[0][-1], color="g", linestyle="dotted", label="Actual concept drift")

plt.legend(loc="upper left")
plt.tight_layout()
plt.xlabel("Window index", fontsize=15)
plt.ylabel("Average NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("figures/single_plot_nmse.pdf", bbox_inches="tight")
