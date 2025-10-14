import numpy as np
from matplotlib import pyplot as plt

with open("../uc_sla_violations_w_cd.npz", "rb") as f:
    pred_sla_violations_w_cd = np.load(f)['arr_0']
    actual_sla_violations = np.load(f)['arr_1']

with open("../uc_sla_violations_wo_cd.npz", "rb") as f:
    pred_sla_violations_wo_cd = np.load(f)['arr_0']

print(sum(pred_sla_violations_w_cd))
print(sum(pred_sla_violations_wo_cd))
print(sum(actual_sla_violations))
print(len(actual_sla_violations) * 100)

plt.subplots_adjust(hspace=0.32)
plt.subplot(2, 1, 1)
plt.title("SLA violation predictions with NDT synchronization")
plt.plot(pred_sla_violations_w_cd, label="Predicted SLA violations")
plt.plot(actual_sla_violations, label="Actual SLA violations")
plt.ylabel("# SLA violations", fontsize=15)
plt.yticks(fontsize=14)
plt.legend()
plt.subplot(2, 1, 2)
plt.title("SLA violation predictions without NDT synchronization")
plt.plot(pred_sla_violations_wo_cd, label="Predicted SLA violations")
plt.plot(actual_sla_violations, label="Actual SLA violations")
plt.legend(loc="upper left")
plt.ylabel("# SLA violations", fontsize=15)
plt.xlabel("Window index", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("all_sla_violations.pdf", bbox_inches='tight')
plt.close()