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

total_operation_time = [418.3, 410, 445, 661.6]

crosshaul_tt = np.array([])
germany_tt = np.array([])
passion_tt = np.array([])
random_tt = np.array([])

for i in range(REALIZATIONS):
    crosshaul_real = np.load(f"{ROOT_DIR}/5g_crosshaul/training_time_5g_crosshaul_{args.target}_r_{i}.npz")["arr_0"]
    germany_real = np.load(f"{ROOT_DIR}/germany/training_time_germany_{args.target}_r_{i}.npz")["arr_0"]
    passion_real = np.load(f"{ROOT_DIR}/passion/training_time_passion_{args.target}_r_{i}.npz")["arr_0"]
    random_real = np.load(f"{ROOT_DIR}/random/training_time_random_{args.target}_r_{i}.npz")["arr_0"]
    if crosshaul_tt.size == 0 and germany_tt.size == 0 and passion_tt.size == 0:
        crosshaul_tt = crosshaul_real
        germany_tt = germany_real
        passion_tt = passion_real
        random_tt = random_real
    else:
        crosshaul_tt = np.vstack([crosshaul_tt, crosshaul_real])
        germany_tt = np.vstack([germany_tt, germany_real])
        passion_tt = np.vstack([passion_tt, passion_real])
        random_tt = np.vstack([random_tt, random_real])

colors = ["#4C72B0", "#55A868", "#C44E52"]
labels = ["After 1st detected drift", "After 2nd detected drift", "After 3rd detected drift"]
plt.subplots_adjust(hspace=0.9)
plt.subplot(4, 1, 1)
plt.title(f"Average retraining time on {args.target} prediction task")
plt.ylim(0, 70)
plt.yticks(fontsize=12)
plt.xlabel("5G-Crosshaul", fontsize=11)

bars = plt.bar(labels, np.mean(crosshaul_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.bar_label(bars, fmt="%0.2f", padding=3, fontweight='bold')

plt.subplot(4, 1, 2)
plt.ylim(0, 70)
plt.yticks(fontsize=12)
plt.xlabel("Germany", fontsize=11)
bars = plt.bar(labels, np.mean(germany_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.bar_label(bars, fmt="%0.2f", padding=3, fontweight='bold')

plt.subplot(4, 1, 3)
plt.ylim(0, 70)
plt.yticks(fontsize=12)
plt.xlabel("PASSION", fontsize=11)
bars = plt.bar(labels, np.mean(passion_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.bar_label(bars, fmt="%0.2f", padding=3, fontweight='bold')

plt.subplot(4, 1, 4)
plt.ylim(0, 140)
plt.yticks(fontsize=12)
plt.xlabel("Synthetic-700", fontsize=11)
bars = plt.bar(labels, np.mean(random_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.bar_label(bars, fmt="%0.2f", padding=3, fontweight='bold')

plt.gcf().supylabel("Retraining time (minutes)", fontsize=13)

plt.savefig(f"figures/training_time_{args.target}.pdf", bbox_inches='tight')

print("=> Numerical info:")
print("Retraining time after [1st, 2nd, 3rd] - 5G-Crosshaul: ", np.mean(crosshaul_tt, axis=0)/60)
print("Retraining time after [1st, 2nd, 3rd] - Germany: ", np.mean(germany_tt, axis=0)/60)
print("Retraining time after [1st, 2nd, 3rd] - PASSION: ", np.mean(passion_tt, axis=0)/60)
print("Retraining time after [1st, 2nd, 3rd] - Synthetic-700: ", np.mean(random_tt, axis=0)/60)

print("\n=> Time spent on retraining (%):")
print("Spent time on 5G-Crosshaul: ",
                            np.sum(np.mean(crosshaul_tt, axis=0)/60)/total_operation_time[0])
print("Spent time on Germany: ",
                            np.sum(np.mean(germany_tt, axis=0)/60)/total_operation_time[1])
print("Spent time on PASSION: ",
                            np.sum(np.mean(passion_tt, axis=0)/60)/total_operation_time[2])
print("Spent time on Synthetic-700: ",
                            np.sum(np.mean(random_tt, axis=0)/60)/total_operation_time[3])
