"""
Script to retrieve NMSE before and after each concept drift event
Created by Cláudio Modesto
LASSE
"""
import numpy as np
import argparse

all_cd = [[0, 68, 131, 194, 251], [0, 72, 143, 211, 246], [0, 72, 135, 198, 267], [0, 95, 203, 293, 397]]
N_REALIZATIONS = 10
topologies = ["5G-Crosshaul", "Germany", "PASSION", "Synthetic-700"]

ROOT_DIR = "../results"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--target", "-g", help="Type of QoS predicted.", 
    type=str, required=True
)

args = parser.parse_args()

nmse_w_drift_5g, nmse_w_drift_ger, \
                nmse_w_drift_passion, nmse_w_drift_random = np.array([]), np.array([]), np.array([]), np.array([])
nmse_wo_drift_5g, nmse_wo_drift_ger, \
                nmse_wo_drift_passion, nmse_w_drift_random = np.array([]), np.array([]), np.array([]), np.array([])

model_updated_region_5g = [[], [], []]
model_updated_region_ger = [[], [], []]
model_updated_region_passion = [[], [], []]
model_updated_region_random = [[], [], []]

for i in range(N_REALIZATIONS):
    if nmse_w_drift_5g.size == 0 and nmse_w_drift_ger.size == 0 and nmse_w_drift_passion.size == 0:
        with open(f"{ROOT_DIR}/5g_crosshaul/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            crosshaul_topology_data = np.load(f)
            nmse_w_drift_5g = crosshaul_topology_data["arr_0"]
            model_updated_region_5g[0].append(crosshaul_topology_data["arr_2"][0])
            model_updated_region_5g[1].append(crosshaul_topology_data["arr_2"][1])
            model_updated_region_5g[2].append(crosshaul_topology_data["arr_2"][2])

        with open(f"{ROOT_DIR}/germany/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            germany_topology_data = np.load(f)
            nmse_w_drift_ger = germany_topology_data["arr_0"]
            model_updated_region_ger[0].append(germany_topology_data["arr_2"][0])
            model_updated_region_ger[1].append(germany_topology_data["arr_2"][1])
            model_updated_region_ger[2].append(germany_topology_data["arr_2"][2])

        with open(f"{ROOT_DIR}/passion/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            passion_topology_data = np.load(f)
            nmse_w_drift_passion = passion_topology_data["arr_0"]
            model_updated_region_passion[0].append(passion_topology_data["arr_2"][0])
            model_updated_region_passion[1].append(passion_topology_data["arr_2"][1])
            model_updated_region_passion[2].append(passion_topology_data["arr_2"][2])

        with open(f"{ROOT_DIR}/random/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            random_topology_data = np.load(f)
            nmse_w_drift_random = random_topology_data["arr_0"]
            model_updated_region_random[0].append(random_topology_data["arr_2"][0])
            model_updated_region_random[1].append(random_topology_data["arr_2"][1])
            model_updated_region_random[2].append(random_topology_data["arr_2"][2])
    else:
        with open(f"{ROOT_DIR}/5g_crosshaul/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_5g = np.vstack([nmse_w_drift_5g, np.load(f)['arr_0']])

        with open(f"{ROOT_DIR}/germany/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_ger = np.vstack([nmse_w_drift_ger, np.load(f)['arr_0']])

        with open(f"{ROOT_DIR}/passion/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_passion = np.vstack([nmse_w_drift_passion, np.load(f)['arr_0']])
        
        with open(f"{ROOT_DIR}/random/results_sync_{args.target}_True_r_{i}.npz", "rb") as f:
            nmse_w_drift_random = np.vstack([nmse_w_drift_random, np.load(f)['arr_0']])

model_updated_regions = [model_updated_region_5g, model_updated_region_ger,
                        model_updated_region_passion, model_updated_region_random]

mean_avg_nmse_5g = np.mean(nmse_w_drift_5g, axis=0)
mean_avg_nmse_ger = np.mean(nmse_w_drift_ger, axis=0)
mean_avg_nmse_passion = np.mean(nmse_w_drift_passion, axis=0)
mean_avg_nmse_random = np.mean(nmse_w_drift_random, axis=0)

with open(f"{ROOT_DIR}/5g_crosshaul/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_5g = np.load(f)['arr_0']

with open(f"{ROOT_DIR}/germany/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_ger = np.load(f)['arr_0']

with open(f"{ROOT_DIR}/passion/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_passion = np.load(f)['arr_0']

with open(f"{ROOT_DIR}/random/results_sync_{args.target}_False_r_0.npz", "rb") as f:
    nmse_wo_drift_random = np.load(f)['arr_0']

all_nmses_w_cd = [mean_avg_nmse_5g, mean_avg_nmse_ger, mean_avg_nmse_passion, mean_avg_nmse_random]

all_nmses_wo_cd = [nmse_wo_drift_5g, nmse_wo_drift_ger, nmse_wo_drift_passion, nmse_wo_drift_random]

print('=> without NDT synchronization:')
for i, nmse in enumerate(all_nmses_wo_cd):
    print(f"{topologies[i]}: {[np.mean(nmse[all_cd[i][j]:all_cd[i][j+1]], axis=0) if j != 0 else np.mean(nmse[all_cd[i][0]:all_cd[i][j+1]], axis=0) for j in range(len(all_cd[i])-1)]}")

print()

print('=> with NDT synchronization:')
for i, nmse in enumerate(all_nmses_w_cd):
    print(f"{topologies[i]}: {[np.mean(nmse[all_cd[i][j]:all_cd[i][j+1]], axis=0) for j in range(len(all_cd[i])-1)]}")
