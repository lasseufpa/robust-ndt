"""
Script to create paper plots (training time)
Created by Cláudio Modesto
LASSE
"""

import os
import argparse
import pathlib
from matplotlib import pyplot as plt
import numpy as np

ROOT_DIR = "../results" # path to root database directory
# Create the output directory
OUTPUT_PATH_NAME = "figures"
if not os.path.isdir(OUTPUT_PATH_NAME):
    pathlib.Path(OUTPUT_PATH_NAME).mkdir(parents=True, exist_ok=True)

REALIZATIONS = 10 # number of training realizations

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target", "-g", help="Type of QoS predicted.", 
    type=str, required=True
)

args = parser.parse_args()

if args.target not in ["delay", "jitter"]:
    raise ValueError("Choose a valid QoS target!")

crosshaul_tt = np.array([])
germany_tt = np.array([])
passion_tt = np.array([])

for i in range(REALIZATIONS):
    crosshaul_real = np.load(f"{ROOT_DIR}/5g_crosshaul/training_time_5g_crosshaul_{args.target}_r_{i}.npz")["arr_0"]
    germany_real = np.load(f"{ROOT_DIR}/germany/training_time_germany_{args.target}_r_{i}.npz")["arr_0"]
    passion_real = np.load(f"{ROOT_DIR}/passion/training_time_passion_{args.target}_r_{i}.npz")["arr_0"]
    if crosshaul_tt.size == 0 and germany_tt.size == 0 and passion_tt.size == 0:
        crosshaul_tt = crosshaul_real
        germany_tt = germany_real
        passion_tt = passion_real
    else:
        crosshaul_tt = np.vstack([crosshaul_tt, crosshaul_real])
        germany_tt = np.vstack([germany_tt, germany_real])
        passion_tt = np.vstack([passion_tt, passion_real])

colors = ["#4C72B0", "#55A868", "#C44E52"]
labels = ["After 1st detected drift", "After 2nd detected drift", "After 3rd detected drift"]
plt.subplots_adjust(hspace=0.55)
plt.subplot(3, 1, 1)
plt.title(f"Average retraining time on {args.target} prediction task")
plt.ylim(0, 50)
plt.xlabel("5G-Crosshaul", fontsize=11)

bars = plt.bar(labels, np.mean(crosshaul_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.bar_label(bars, fmt="%0.2f", padding=3, fontweight='bold')

plt.subplot(3, 1, 2)
plt.ylim(0, 50)
plt.xlabel("Germany", fontsize=11)
plt.ylabel("Retraining time (minutes)", fontsize=12)
bars = plt.bar(labels, np.mean(germany_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.bar_label(bars, fmt="%0.2f", padding=3, fontweight='bold')

plt.subplot(3, 1, 3)
plt.ylim(0, 50)
plt.xlabel("PASSION", fontsize=11)
bars = plt.bar(labels, np.mean(passion_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.bar_label(bars, fmt="%0.2f", padding=3, fontweight='bold')

plt.savefig(f"figures/training_time_{args.target}.pdf", bbox_inches='tight')

print("=> Numerical info:")
print("Retraining time after [1st, 2nd, 3rd]: ", np.mean(crosshaul_tt, axis=0)/60)
print("Retraining time after [1st, 2nd, 3rd]: ", np.mean(germany_tt, axis=0)/60)
print("Retraining time after [1st, 2nd, 3rd]: ", np.mean(passion_tt, axis=0)/60)
