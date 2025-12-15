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

parser.add_argument(
    "--t-topology", "-t", help="Topology results that should be appear on the top of the plot [5g_crosshaul | germany | passion | random].", 
    type=str, required=True
)

parser.add_argument(
    "--b-topology", "-b", help="Topology results that should be appear on the bottom of the plot [5g_crosshaul | germany | passion | random].", 
    type=str, required=True
)

args = parser.parse_args()

ROOT_DIR = "../results"
all_cd = [[68, 131, 194], [72, 143, 211], [72, 135, 198], [95, 203, 293]]

title_name = ""
if args.t_topology == "5g_crosshaul" or args.b_topology == "5g_crosshaul":
    if args.t_topology == "5g_crosshaul":
        top_title_name = "5G-Crosshaul"
        top_concept_drift = all_cd[0]
    elif args.b_topology == "5g_crosshaul":
        bottom_title_name = "5G-Crosshaul"
        bottom_concept_drift = all_cd[0]
if args.t_topology == "germany" or args.b_topology == "germany":
    if args.t_topology == "germany":
        top_title_name = "Germany"
        top_concept_drift = all_cd[1]
    elif args.b_topology == "germany":
        bottom_title_name = "Germany"
        bottom_concept_drift = all_cd[1]
if args.t_topology == "passion" or args.b_topology == "passion":
    if args.t_topology == "passion":
        top_title_name = "PASSION"
        top_concept_drift = all_cd[2]
    elif args.b_topology == "passion":
        bottom_title_name = "PASSION"
        bottom_concept_drift = all_cd[2]
if args.t_topology == "random" or args.b_topology == "random":
    if args.t_topology == "random":
        top_title_name = "Synthetic-700"
        top_concept_drift = all_cd[3]
    elif args.b_topology == "random":
        bottom_title_name = "Synthetic-700"
        bottom_concept_drift = all_cd[3]

N_REALIZATIONS = 10
nmse_w_drift_top = np.array([])
nmse_w_drift_bottom = np.array([])

model_updated_region_top = [[], [], []]
model_updated_region_bottom = [[], [], []]
for i in range(N_REALIZATIONS):
    with open(f"{ROOT_DIR}/{args.t_topology}/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
        top_data = np.load(f)
        if nmse_w_drift_top.size == 0:
            nmse_w_drift_top = top_data["arr_0"]
        else:
            nmse_w_drift_top = np.vstack([nmse_w_drift_top, top_data["arr_0"]])
        if i == 0:
            drift_detected_top = top_data["arr_1"]
        model_updated_region_top[0].append(top_data["arr_2"][0])
        model_updated_region_top[1].append(top_data["arr_2"][1])
        model_updated_region_top[2].append(top_data["arr_2"][2])

    with open(f"{ROOT_DIR}/{args.b_topology}/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
        bottom_data = np.load(f)
        if nmse_w_drift_bottom.size == 0:
            nmse_w_drift_bottom = bottom_data["arr_0"]
        else:
            nmse_w_drift_bottom = np.vstack([nmse_w_drift_bottom, bottom_data["arr_0"]])
        if i == 0:
            drift_detected_bottom = np.load(f)["arr_1"]
        model_updated_region_bottom[0].append(bottom_data["arr_2"][0])
        model_updated_region_bottom[1].append(bottom_data["arr_2"][1])
        model_updated_region_bottom[2].append(bottom_data["arr_2"][2])

with open(f"{ROOT_DIR}/{args.t_topology}/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_top = np.load(f)["arr_0"]

with open(f"{ROOT_DIR}/{args.b_topology}/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_bottom = np.load(f)["arr_0"]

plt.figure(figsize=(9, 7))

plt.subplot(2, 1, 1)
plt.ylim(-45, 55)
plt.title(f"{args.target.title()} prediction performance in {top_title_name}", fontsize=17)
plt.yticks([-40, -20, 0, 20, 40])
avg_nmse_w_drift_top = np.mean(nmse_w_drift_top, axis=0)
std_nmse_w_drift_top = np.std(nmse_w_drift_top, axis=0)
plt.plot(avg_nmse_w_drift_top, label="VTwin w/ retraining", linewidth=2)
plt.plot(nmse_wo_drift_top, label="VTwin w/o retraining", color="k", alpha=0.6, linestyle="--")
plt.fill_between(np.arange(avg_nmse_w_drift_top.size), avg_nmse_w_drift_top - std_nmse_w_drift_top,
                        avg_nmse_w_drift_top + std_nmse_w_drift_top,
                        color='blue', alpha=0.25, label='Standard deviation')
for index in drift_detected_top[:-1]:
    plt.axvline(x=index-1, color="r", linestyle="dotted")
plt.axvline(x=drift_detected_top[-1]-1, color="r",
                                linestyle="dotted", label="Concept drift detected")

for i in range(len(model_updated_region_top)):
    plt.axvspan(min(model_updated_region_top[i]), max(model_updated_region_top[i]), alpha=0.3, color="red")

for i in range(len(top_concept_drift) - 1):
    plt.axvline(x=top_concept_drift[i], color='g', linestyle='dotted')
plt.axvline(x=top_concept_drift[-1], color='g', linestyle='dotted', label='Actual concept drift')
plt.legend(loc="upper left")
plt.ylabel("Average NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

avg_nmse_w_drift_bottom = np.mean(nmse_w_drift_bottom, axis=0)
std_nmse_w_drift_bottom = np.std(nmse_w_drift_bottom, axis=0)

plt.subplot(2, 1, 2)
plt.plot(avg_nmse_w_drift_bottom, label="VTwin w/ retraining", linewidth=2)
plt.plot(nmse_wo_drift_bottom, label="VTwin w/o retraining", color="k", alpha=0.6, linestyle="--")
plt.fill_between(np.arange(avg_nmse_w_drift_bottom.size), avg_nmse_w_drift_bottom - std_nmse_w_drift_bottom,
                        avg_nmse_w_drift_bottom + std_nmse_w_drift_bottom,
                        color='blue', alpha=0.25, label='Standard deviation')
plt.title(f"{args.target.title()} prediction performance in {bottom_title_name}", fontsize=17)
plt.tight_layout()
plt.ylim(-45, 55)
for index in drift_detected_bottom[:-1]:
    plt.axvline(x=index-1, color="r", linestyle="dotted")
plt.axvline(x=drift_detected_bottom[-1]-1, color="r", linestyle="dotted", label="Concept drift detected")

for i in range(len(model_updated_region_bottom)):
    plt.axvspan(min(model_updated_region_bottom[i]), max(model_updated_region_bottom[i]), alpha=0.3, color="red")

for i in range(len(bottom_concept_drift) - 1):
    plt.axvline(x=bottom_concept_drift[i], color="g", linestyle="dotted")
plt.axvline(x=bottom_concept_drift[-1], color="g", linestyle="dotted", label="Actual concept drift")

plt.ylabel("Average NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.xlabel("Window index", fontsize=15)
plt.ylabel("Average NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(f"figures/multiple_plots_{args.target}_{args.t_topology}_{args.b_topology}_nmse.pdf", bbox_inches="tight")
