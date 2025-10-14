import numpy as np
from matplotlib import pyplot as plt

all_cd = [[68, 131, 194], [72, 143, 211], [72, 135, 198]]

with open("../results/mapes_with_admin_True_passion.npz", "rb") as f:
    nmse_w_drift = np.load(f)["arr_0"]
    drift_detected = np.load(f)["arr_1"]
    model_updated = np.load(f)["arr_2"]

with open("../results/mapes_with_admin_True_germany.npz", "rb") as f:
    nmse_w_drift_ger = np.load(f)["arr_0"]
    drift_detected_ger = np.load(f)["arr_1"]
    model_updated_ger = np.load(f)["arr_2"]

with open("../results/mapes_with_admin_False_passion.npz", "rb") as f:
    nmse_wo_drift = np.load(f)["arr_0"]

with open("../results/mapes_with_admin_False_germany.npz", "rb") as f:
    nmse_wo_drift_ger = np.load(f)["arr_0"]

plt.figure(figsize=(8, 6)) 
plt.subplot(2, 1, 1)
plt.plot(nmse_w_drift, label="VTwin w/ retraining", linewidth=2)
plt.plot(nmse_wo_drift, label="VTwin w/o retraining", color="k", alpha=0.6, linestyle="--")
plt.title("Germany")
plt.tight_layout()
plt.ylim(-45, 55)
for index in drift_detected[:-1]:
    plt.axvline(x=index-1, color="r", linestyle="dotted")
plt.axvline(x=drift_detected[-1]-1, color="r", linestyle="dotted", label="Concept drift detected")
for index in model_updated[:-1]:
    plt.scatter(index, nmse_w_drift[index], marker="x", color="red", zorder=2, s=55)
plt.scatter(model_updated[-1], nmse_w_drift[-1], marker="x", color="red", zorder=2, s=55, label="Model updated")
for i in range(len(all_cd[1]) - 1):
    plt.axvline(x=all_cd[1][i], color="g", linestyle="dotted")
plt.axvline(x=all_cd[1][-1], color="g", linestyle="dotted", label="Actual concept drift")

plt.legend(loc="upper left")
plt.ylabel("NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.subplot(2, 1, 2)
plt.ylim(-45, 55)
plt.title("PASSION")
plt.yticks([-40, -20, 0, 20, 40])
plt.plot(nmse_w_drift_ger, label="VTwin w/ retraining", linewidth=2)
plt.plot(nmse_wo_drift_ger, label="VTwin w/o retraining", color="k", alpha=0.6, linestyle="--")
for index in drift_detected_ger[:-1]:
    plt.axvline(x=index-1, color="r", linestyle="dotted")
plt.axvline(x=drift_detected_ger[-1]-1, color="r", linestyle="dotted", label="Concept drift detected")
for index in model_updated[:-1]:
    plt.scatter(index, nmse_w_drift_ger[index], marker="x", color="red", zorder=2, s=55)
plt.scatter(model_updated[-1], nmse_w_drift_ger[-1], marker="x", color="red", zorder=2, s=55, label="Model updated")
for i in range(len(all_cd[2]) - 1):
    plt.axvline(x=all_cd[2][i], color='g', linestyle='dotted')
plt.axvline(x=all_cd[2][-1], color='g', linestyle='dotted', label='Actual concept drift')

plt.tight_layout()
plt.xlabel("Window index", fontsize=15)
plt.ylabel("NMSE (dB)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("NMSE_germany_passion.pdf", bbox_inches="tight")
