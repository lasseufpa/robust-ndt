"""
Main script to generate traffic using iperf traffic generator
Created by: Cleverson Nahum
Edited by: Cláudio Modesto
LASSE
"""
import os
import pathlib
import argparse
import shutil
import json
import itertools
import numpy as np
import networkx as nx
from mininet.log import setLogLevel
from network_scenario import Network

def create_flows_description(number_nodes, net_topology, experiment_dir) -> dict:
    """
    Create the description of traffic generation
    including the number of connections and streams
    """
    n_conn = 80 # number of server-client pairs
    n_streams = 1
    duration = 9000
    pattern = "pareto"
    packet_size = 512
    rng = np.random.default_rng()
    flows_description = {}

    all_src_dst = list(itertools.combinations(range(number_nodes), 2))
    src_dst_idx = rng.choice(len(all_src_dst), n_conn, replace=False)

    selected_edges = []
    selected_nodes = []
    for conn_idx in range(n_conn):
        sp = nx.shortest_path(net_topology,
                            source=str(all_src_dst[src_dst_idx[conn_idx]][0]),
                            target=str(all_src_dst[src_dst_idx[conn_idx]][1]))
        current_edges = [[sp[i], sp[i+1]] for i in range(len(sp[:-1]))]
        use_edge = [x in selected_edges for x in current_edges]
        if any(use_edge) or all_src_dst[src_dst_idx[conn_idx]][1] in selected_nodes:
            continue
        selected_edges += current_edges
        selected_nodes.append(all_src_dst[src_dst_idx[conn_idx]][1])
        flows_description[f"conn_{conn_idx}"] = {
            "src": all_src_dst[src_dst_idx[conn_idx]][0],
            "dst": all_src_dst[src_dst_idx[conn_idx]][1],
            "n_streams": n_streams,
            "pattern": pattern,
            "packet_size": packet_size,
            "duration": duration,
            "conn_id": conn_idx
        }

    traffic_metadata = {}
    traffic_metadata["tr_metadata"] = {"max_n_conn": n_conn,
                                    "topology": net_topology.graph.get("label", "no label"),
                                    "n_streams": n_streams,
                                    "packet_size": packet_size,
                                    "duration": duration,
                                    "pattern": pattern}
    all_mininet_metadata = {}
    mininet_data_filename = f"{experiment_dir}/mininet_data.json"
    if os.path.isfile(mininet_data_filename):
        with open(mininet_data_filename, "r", encoding="utf-8") as f:
            all_mininet_metadata = json.load(f)
    if "tr_metadata" in all_mininet_metadata.keys():
        all_mininet_metadata["tr_metadata"].update(traffic_metadata)
    else:
        all_mininet_metadata["tr_metadata"] = traffic_metadata

    with open(f"{experiment_dir}/mininet_data.json", "w", encoding="utf-8") as f:
        json.dump(all_mininet_metadata, f, indent=4)

    return flows_description

def main(identifier: int, topo_filepath: str):
    """
    main script for generate traffic
    """
    setLogLevel("info")
    topo_params = {
        "bandwidth": "capacity",
        "delay": "delay",
        "loss": "loss",
    }

    experiment_dir = f"./logs/experiment_{identifier}"

    # seconds (if it is too small, the topology may not be available in the ONOS yet)
    time_wait_topology = 20

    # seconds (if it is too small, the flows may not be available in the ONOS yet)
    # Create folder and log file
    pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(topo_filepath, f"{experiment_dir}/topology.gml")
    net_topology = nx.read_gml(f"{experiment_dir}/topology.gml")
    network = Network(topo_file=topo_filepath, topo_params=topo_params)
    flows_description = create_flows_description(len(network.net.hosts),
                                                net_topology,
                                                experiment_dir)
    print("Flows description:\n", flows_description)
    network.start(
        time_wait_topology=time_wait_topology,
        flows_description=flows_description,
        experiment_dir=experiment_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the traffic generator")
    parser.add_argument("--id", type=int, required=True, help="Experiment identifier")
    parser.add_argument("--topo-filepath",
                        type=str,
                        required=True,
                        help="Network topology .gml file")
    args = parser.parse_args()

    main(args.id, args.topo_filepath)
