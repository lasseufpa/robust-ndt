''' 
Main script to generate traffic using iperf traffic generator
Created by: Cleverson Nahum
Edited by: Cláudio Modesto
'''
import argparse
import pathlib
import shutil
import itertools
import numpy as np
from mininet.log import setLogLevel
from network_scenario import Network

def create_flows_description(number_nodes) -> dict:
    '''
    Create the description of traffic generation
    including the number of connections and streams
    '''
    n_conn = 30 # number of server-client pairs
    n_streams = 1 # 4, 10, 5, 9
    bandwidth = 20 # Mbits/s
    duration = 5000 # traffic duration time in seconds
    rng = np.random.default_rng()
    flows_description = {}

    all_src_dst = list(itertools.combinations(range(number_nodes), 2))
    src_dst_idx = rng.choice(len(all_src_dst), n_conn, replace=False)

    selected_nodes = []
    for conn_idx in range(n_conn):
        if all_src_dst[src_dst_idx[conn_idx]][1] in selected_nodes:
            continue
        selected_nodes.append(all_src_dst[src_dst_idx[conn_idx]][1])
        flows_description[f"conn_{conn_idx}"] = {
            "src": all_src_dst[src_dst_idx[conn_idx]][0],
            "dst": all_src_dst[src_dst_idx[conn_idx]][1],
            "n_streams": n_streams,
            "bandwidth": bandwidth,
            "duration": duration, # seconds 8000, 500 (NSFNet), 600 (GBN)
            "conn_id": conn_idx
        }

    return flows_description


def main(identifier: int, topo_filepath: str):
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
    
    network = Network(topo_file=topo_filepath, topo_params=topo_params)
    flows_description = create_flows_description(len(network.net.hosts))
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
