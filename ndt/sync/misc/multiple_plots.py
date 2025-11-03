"""
Script to create paper multiple plots (NMSE)
Created by Cláudio Modesto
LASSE
"""

import os
import argparse
import pathlib
import numpy as np
from matplotlib import pyplot as plt

ROOT_DIR = "../results" # path to root database directory
# Create the output directory
OUTPUT_PATH_NAME = "figures"
if not os.path.isdir(OUTPUT_PATH_NAME):
    pathlib.Path(OUTPUT_PATH_NAME).mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--target", "-g", help="Type of QoS predicted.", 
    type=str, required=True
)

args = parser.parse_args()

ROOT_DIR = "../results"
all_cd = [[68, 131, 194], [72, 143, 211], [72, 135, 198]]

N_REALIZATIONS = 10
nmse_w_drift_ger = np.array([])
nmse_w_drift_passion = np.array([])

for i in range(N_REALIZATIONS):
    with open(f"{ROOT_DIR}/passion/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
        if nmse_w_drift_passion.size == 0:
            nmse_w_drift_passion = np.load(f)["arr_0"]
        else:
            nmse_w_drift_passion = np.vstack([nmse_w_drift_passion, np.load(f)["arr_0"]])
        if i == 0:
            drift_detected_passion = np.load(f)["arr_1"]
            model_updated_passion = np.load(f)["arr_2"]

    with open(f"{ROOT_DIR}/germany/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
        if nmse_w_drift_ger.size == 0:
            nmse_w_drift_ger = np.load(f)["arr_0"]
        else:
            nmse_w_drift_ger = np.vstack([nmse_w_drift_ger, np.load(f)["arr_0"]])
        if i == 0:
            drift_detected_ger = np.load(f)["arr_1"]
            model_updated_ger = np.load(f)["arr_2"]

with open(f"{ROOT_DIR}/passion/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_passion = np.load(f)["arr_0"]

with open(f"{ROOT_DIR}/germany/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_ger = np.load(f)["arr_0"]

plt.figure(figsize=(9, 7))

plt.subplot(2, 1, 1)
plt.ylim(-45, 55)
plt.title(f"{args.target.title()} prediction performance in Germany", fontsize=17)
plt.yticks([-40, -20, 0, 20, 40])
avg_nmse_w_drift_ger = np.mean(nmse_w_drift_ger, axis=0)
std_nmse_w_drift_ger = np.std(nmse_w_drift_ger, axis=0)
plt.plot(avg_nmse_w_drift_ger, label="VTwin w/ retraining", linewidth=2)
plt.plot(nmse_wo_drift_ger, label="VTwin w/o retraining", color="k", alpha=0.6, linestyle="--")
plt.fill_between(np.arange(avg_nmse_w_drift_ger.size), avg_nmse_w_drift_ger - std_nmse_w_drift_ger,
                        avg_nmse_w_drift_ger + std_nmse_w_drift_ger,
                        color='blue', alpha=0.25, label='Standard deviation')
for index in drift_detected_ger[:-1]:
    plt.axvline(x=index-1, color="r", linestyle="dotted")
plt.axvline(x=drift_detected_ger[-1]-1, color="r",
                                linestyle="dotted", label="Concept drift detected")
for index in model_updated_ger[:-1]:
    plt.scatter(index, avg_nmse_w_drift_ger[index],
                                            marker="x", color="red", zorder=2, s=55)
plt.scatter(model_updated_ger[-1], avg_nmse_w_drift_ger[-1],
                        marker="x", color="red", zorder=2, s=55, label="Model updated")
for i in range(len(all_cd[1]) - 1):
    plt.axvline(x=all_cd[1][i], color='g', linestyle='dotted')
plt.axvline(x=all_cd[1][-1], color='g', linestyle='dotted', label='Actual concept drift')
plt.legend(loc="upper left")
plt.ylabel("Average NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

avg_nmse_w_drift_passion = np.mean(nmse_w_drift_passion, axis=0)
std_nmse_w_drift_passion = np.std(nmse_w_drift_passion, axis=0)

plt.subplot(2, 1, 2)
plt.plot(avg_nmse_w_drift_passion, label="VTwin w/ retraining", linewidth=2)
plt.plot(nmse_wo_drift_passion, label="VTwin w/o retraining", color="k", alpha=0.6, linestyle="--")
plt.fill_between(np.arange(avg_nmse_w_drift_passion.size), avg_nmse_w_drift_passion - std_nmse_w_drift_passion,
                        avg_nmse_w_drift_passion + std_nmse_w_drift_passion,
                        color='blue', alpha=0.25, label='Standard deviation')
plt.title(f"{args.target.title()} prediction performance in PASSION", fontsize=17)
plt.tight_layout()
plt.ylim(-45, 55)
for index in drift_detected_passion[:-1]:
    plt.axvline(x=index-1, color="r", linestyle="dotted")
plt.axvline(x=drift_detected_passion[-1]-1, color="r", linestyle="dotted", label="Concept drift detected")
for index in model_updated_passion[:-1]:
    plt.scatter(index, avg_nmse_w_drift_passion[index],
                                        marker="x", color="red", zorder=2, s=55)
plt.scatter(model_updated_passion[-1], nmse_wo_drift_passion[-1], marker="x", color="red", zorder=2, s=55, label="Model updated")
for i in range(len(all_cd[2]) - 1):
    plt.axvline(x=all_cd[2][i], color="g", linestyle="dotted")
plt.axvline(x=all_cd[2][-1], color="g", linestyle="dotted", label="Actual concept drift")

plt.ylabel("Average NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.xlabel("Window index", fontsize=15)
plt.ylabel("Average NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"figures/multiple_plots_{args.target}_nmse.pdf", bbox_inches="tight")
