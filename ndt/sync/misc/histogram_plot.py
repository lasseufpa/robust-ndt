"""
Script to create paper histogram plots
of the network characteristics
Created by Cláudio Modesto
"""

import networkx as nx
from matplotlib import pyplot as plt

crosshaul = nx.read_gml("../traffic_generator/topology/5G_crosshaul_51.gml")
germany = nx.read_gml("../traffic_generator/topology/germany_50.gml")
passion = nx.read_gml("../traffic_generator/topology/HPASSION_128.gml")

crosshaul_capacity = nx.get_edge_attributes(crosshaul, "capacity")
crosshaul_delay = nx.get_edge_attributes(crosshaul, "delay")

germany_capacity = nx.get_edge_attributes(germany, "capacity")
germany_delay = nx.get_edge_attributes(germany, "delay")

passion_capacity = nx.get_edge_attributes(passion, "capacity")
passion_delay = nx.get_edge_attributes(passion, "delay")

plt.subplot(3, 1, 1)
plt.hist(crosshaul_capacity.values(), color='#4C72B0', edgecolor='white', alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("5G-Crosshaul")
plt.subplot(3, 1, 2)
plt.hist(germany_capacity.values(), color='#4C72B0', edgecolor='white', alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Occurrences", fontsize=14)
plt.title("Germany")
plt.subplot(3, 1, 3)
plt.hist(passion_capacity.values(), color='#4C72B0', edgecolor='white', alpha=0.8)
plt.tight_layout()
plt.xlabel("Capacity (Mbits/s)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("PASSION")
plt.savefig("Hist_capacities.pdf", bbox_inches="tight")
plt.close()

plt.subplot(3, 1, 1)
plt.hist(crosshaul_delay.values(), color='#069736', edgecolor='white', alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("5G-Crosshaul")
plt.subplot(3, 1, 2)
plt.hist(germany_delay.values(), color='#069736', edgecolor='white', alpha=0.8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Occurrences", fontsize=14)
plt.title("Germany")
plt.subplot(3, 1, 3)
plt.hist(passion_delay.values(), color="#069736", edgecolor='white', alpha=0.8)
plt.tight_layout()
plt.xlabel("Propagation delay (s)", fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("PASSION")
plt.savefig("Hist_delay.pdf", bbox_inches="tight")
plt.close()
