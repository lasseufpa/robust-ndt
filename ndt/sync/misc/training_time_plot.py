"""
Script to create paper plots (training time)
Created by Cláudio Modesto
"""

import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np

ROOT_DIR = "../results" # path to root database directory
# Create the output directory
OUTPUT_PATH_NAME = "figures"
if not os.path.isdir(OUTPUT_PATH_NAME):
    pathlib.Path(OUTPUT_PATH_NAME).mkdir(parents=True, exist_ok=True)

REALIZATIONS = 10 # number of training realizations

crosshaul_tt = np.array([])
germany_tt = np.array([])
passion_tt = np.array([])

for i in range(REALIZATIONS):
    crosshaul_real = np.load(f"{ROOT_DIR}/5g_crosshaul/training_time_5g_crosshaul_r_{i}.npz")["arr_0"]
    germany_real = np.load(f"{ROOT_DIR}/germany/training_time_germany_r_{i}.npz")["arr_0"]
    passion_real = np.load(f"{ROOT_DIR}/passion/training_time_passion_r_{i}.npz")["arr_0"]
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
plt.ylim(0, 40)
plt.title("5G-Crosshaul")
plt.bar(labels, np.mean(crosshaul_tt, axis=0)/60, 
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.subplot(3, 1, 2)
plt.ylim(0, 40)
plt.title("Germany")
plt.ylabel("Retraining time (minutes)", fontsize=15)
plt.bar(labels, np.mean(germany_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.subplot(3, 1, 3)
plt.ylim(0, 40)
plt.title("PASSION")
plt.bar(labels, np.mean(passion_tt, axis=0)/60,
                    yerr=np.std(crosshaul_tt, axis=0)/60, capsize=5, width=0.4, color=colors)
plt.xlabel("Concept drift event", fontsize=15)
plt.savefig("figures/training_time.pdf", bbox_inches='tight')

print("=> Numerical info:")
print("Retraining time after [1st, 2nd, 3rd]: ", np.mean(crosshaul_tt, axis=0)/60)
print("Retraining time after [1st, 2nd, 3rd]: ", np.mean(germany_tt, axis=0)/60)
print("Retraining time after [1st, 2nd, 3rd]: ", np.mean(passion_tt, axis=0)/60)
