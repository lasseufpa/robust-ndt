"""
Script to create paper plots (accuracy)
Created by Cláudio Modesto
LASSE
"""

import os
import argparse
import pathlib
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--topology", "-t", help="Type of topology to be used in the experiments.", 
    type=str, required=True
)

args = parser.parse_args()

all_cd = [[0, 68, 131, 194, 251], [0, 72, 143, 211, 246], [0, 72, 135, 198, 267]]

ROOT_DIR = "../results" # path to root database directory
# Create the output directory
OUTPUT_PATH_NAME = "figures"
if not os.path.isdir(OUTPUT_PATH_NAME):
    pathlib.Path(OUTPUT_PATH_NAME).mkdir(parents=True, exist_ok=True)

ROOT_DIR = "../results"

VALID_TOPOLOGY = False
if (args.topology == "5g_crosshaul" or args.topology == "germany" or
                    args.topology == "passion"):
    VALID_TOPOLOGY = True
else:
    raise ValueError("Choose a valid topology!")

if VALID_TOPOLOGY:
    with open(f"{ROOT_DIR}/{args.topology}/uc_violations_True_r_0.npz", "rb") as f:
        pred_sla_violations_w_cd = np.load(f)['arr_0']
        correct_pred_w = np.load(f)['arr_2']
        new_model = np.load(f)['arr_4']

    with open(f"{ROOT_DIR}/{args.topology}/uc_violations_False_r_0.npz", "rb") as f:
        correct_pred_wo = np.load(f)['arr_2']

    SAMPLES = 100
    accuracy = (correct_pred_w/SAMPLES) * 100

    plt.figure(figsize=(9, 7))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.title("SLA violation predictions with NDT synchronization", fontsize=17)
    plt.plot(accuracy, label="Accuracy")
    plt.scatter(all_cd[0][1:-1], [accuracy[acc-1] for acc in all_cd[0][1:-1]],
            marker='x', color='red', zorder=2, s=55,
            label="Concept drift")
    plt.scatter(new_model, [accuracy[update] for update in new_model],
            marker='x', color='black', zorder=2, s=55,
            label="New VTwin")

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title("SLA violation predictions without NDT synchronization", fontsize=17)
    plt.plot((correct_pred_wo/SAMPLES) * 100, label="Accuracy", color="green")
    plt.xlabel("Window index", fontsize=15)
    plt.ylabel("Accuracy (%)", fontsize=15)
    plt.scatter(all_cd[0][1:-1], [accuracy[acc-1] for acc in all_cd[0][1:-1]],
            marker='x', color='red', zorder=2, s=55,
            label="Concept drift")
    plt.legend()
    plt.savefig(f"figures/accuracy_plot_{args.topology}.pdf", bbox_inches="tight")
    