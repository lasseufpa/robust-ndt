"""
Script to retrieve NMSE before and after each concept drift event
Created by Cláudio Modesto
"""
import numpy as np

all_cd = [[0, 68, 131, 194, 251], [0, 72, 143, 211, 246], [0, 72, 135, 198, 267]]
N_REALIZATIONS = 10
topologies = ["5G-Crosshaul", "Germany", "PASSION"]

ROOT_DIR = "../results"

nmse_w_drift_5g, nmse_w_drift_ger, nmse_w_drift_passion = np.array([]), np.array([]), np.array([])
nmse_wo_drift_5g, nmse_wo_drift_ger, nmse_wo_drift_passion = np.array([]), np.array([]), np.array([])

for i in range(N_REALIZATIONS):
    if nmse_w_drift_5g.size == 0 and nmse_w_drift_ger.size == 0 and nmse_w_drift_passion.size == 0:
        with open(f"{ROOT_DIR}/5g_crosshaul/results_sync_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_5g = np.load(f)['arr_0']

        with open(f"{ROOT_DIR}/germany/results_sync_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_ger = np.load(f)['arr_0']

        with open(f"{ROOT_DIR}/passion/results_sync_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_passion = np.load(f)['arr_0']
    else:
        with open(f"{ROOT_DIR}/5g_crosshaul/results_sync_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_5g = np.vstack([nmse_w_drift_5g, np.load(f)['arr_0']])

        with open(f"{ROOT_DIR}/germany/results_sync_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_ger = np.vstack([nmse_w_drift_ger, np.load(f)['arr_0']])

        with open(f"{ROOT_DIR}/passion/results_sync_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_passion = np.vstack([nmse_w_drift_passion, np.load(f)['arr_0']])

mean_avg_nmse_5g = np.mean(nmse_w_drift_5g, axis=0)
mean_avg_nmse_ger = np.mean(nmse_w_drift_ger, axis=0)
mean_avg_nmse_passion = np.mean(nmse_w_drift_passion, axis=0)

with open(f"{ROOT_DIR}/5g_crosshaul/results_sync_False_r_0.npz", "rb") as f:
    nmse_wo_drift_5g = np.load(f)['arr_0']

with open(f"{ROOT_DIR}/germany/results_sync_False_r_0.npz", "rb") as f:
    nmse_wo_drift_ger = np.load(f)['arr_0']

with open(f"{ROOT_DIR}/passion/results_sync_False_r_0.npz", "rb") as f:
    nmse_wo_drift_passion = np.load(f)['arr_0']

all_nmses_w_cd = [mean_avg_nmse_5g, mean_avg_nmse_ger, mean_avg_nmse_passion]

all_nmses_wo_cd = [nmse_wo_drift_5g, nmse_wo_drift_ger, nmse_wo_drift_passion]

print('=> with NDT synchronization:')
for i, nmse in enumerate(all_nmses_w_cd):
    print(f"{topologies[i]}: {[np.mean(nmse[all_cd[i][j-1]:all_cd[i][j]], axis=0) for j in range(1, len(all_cd[i]))]}")

print()
print('=> without NDT synchronization:')
for i, nmse in enumerate(all_nmses_wo_cd):
    print(f"{topologies[i]}: {[np.mean(nmse[all_cd[i][j-1]:all_cd[i][j]]) for j in range(1, len(all_cd[i]))]}")
