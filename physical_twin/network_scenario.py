"""
Script to instantiate a Mininet topology and generate traffic
Created by: Cleverson Nahum
Edited by: Cláudio Modesto
LASSE
"""

import os
import json
from time import sleep
from typing import Optional

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.node import Host
from mininet.node import OVSKernelSwitch
from mininet.log import info
from mininet.link import TCLink
import networkx as nx
import requests


class Network:
    """
    Define mininet topology methods and attributes 
    """
    def __init__(
        self,
        topo_file: str,
        topo_params: Optional[dict] = None,
        onos_addr: str = "127.0.0.1",
        onos_port: int = 6653,
        onos_api_port: int = 8181,
        onos_user: str = "onos",
        onos_pass: str = "rocks",
    ):
        self.net_topology = nx.read_gml(topo_file)
        self.net = Mininet(
            controller=RemoteController, switch=OVSKernelSwitch, link=TCLink
        )
        self.onos_api_adress = f"http://{onos_addr}:{onos_api_port}/onos/v1"
        self.onos_user = onos_user
        self.onos_pass = onos_pass
        self.net.addController(
            name="c0",
            controller=RemoteController,
            ip=onos_addr,
            protocol="tcp",
            port=onos_port,
        )
        self.switches = []
        self.hosts = []
        if topo_params is None:
            topo_params = {
                "bandwidth": "bandwidth",
                "delay": "delay",
                "loss": "loss",
            }
        link_default_values = {
            "bandwidth": 100,
            "delay": "1ms",
            "loss": 0,
            "queue_size": 1024
        }

        # Creating switches and hosts
        for node in self.net_topology.nodes:
            idx = int(node) + 1
            # Switch
            curr_switch = self.net.addSwitch(
                f"s{idx}",
                cls=OVSKernelSwitch,
                protocols="OpenFlow10",
                mac_address=self.gen_mac_address(idx),
            )
            self.switches.append(curr_switch)
            # Host
            curr_host = self.net.addHost(
                f"h{idx}",
                cls=Host,
            )
            self.hosts.append(curr_host)
            # Creating links between switch and host
            self.net.addLink(self.hosts[int(idx) - 1], self.switches[int(idx) - 1])

        # Creating switch links
        for edge in self.net_topology.edges(data=True):
            src = int(edge[0])
            dst = int(edge[1])
            bw = edge[2].get(topo_params["bandwidth"], link_default_values["bandwidth"])
            delay = edge[2].get(topo_params["delay"], link_default_values["delay"])
            loss = edge[2].get(topo_params["loss"], link_default_values["loss"])
            self.net.addLink(
                self.switches[src],
                self.switches[dst],
                bw=bw,
                delay=f"{delay}ms",
                loss=loss,
            )

    def start(
        self,
        time_wait_topology: int,
        flows_description: dict,
        experiment_dir: str = "./logs",
    ):
        """
        Instatiate a miniet topology and generate traffic in it
        """
        self.net.start()
        info("INFO: Mininet topology started\n")
        sleep(time_wait_topology)
        info("INFO: Traffic generator started\n")
        self.start_servers(flows_description)
        self.start_clients(flows_description, experiment_dir)
        info(f"INFO: Experiment is finishing, wait a second. You can find log information in {experiment_dir}\n")
        sleep(10)
        self.net.stop()

    def gen_mac_address(self, idx: int):
        """
        Get MAC address of a given network switch by id
        """
        return f"00:00:00:00:00:{idx:02x}"

    def get_device_id(self, idx: int):
        """
        Get device id from openflow hexidecimal identifier
        """
        return f"of:{(idx+1):016d}"

    def start_servers(self, flows_description: dict):
        """
        Starts mgen server
        """
        for _, conn_info in flows_description.items():
            mn_dst_host = self.hosts[conn_info["dst"]]
            mn_dst_host.cmd("ITGRecv&")

    def start_clients(self,
                    flows_description: dict,
                    experiment_dir: str = "./logs"):
        """
        Starts ditg client
        """
        flowpath_filename = f"{experiment_dir}/conn_paths.json"
        if os.path.isfile(flowpath_filename):
            os.remove(flowpath_filename)

        duration = 0
        for _, conn_info in flows_description.items():
            mn_src_host = self.hosts[conn_info["src"]]
            dst_host = conn_info["dst"] + 1
            conn_id = conn_info["conn_id"]
            packet_size = conn_info["packet_size"]
            shortest_path = nx.shortest_path(self.net_topology,
                            source=str(conn_info["src"]),
                            target=str(conn_info["dst"]))
            print(shortest_path)
            all_capacities = [self.net_topology.get_edge_data(
                    str(shortest_path[i]), str(shortest_path[i+1]))[0].get("capacity")
                    for i in range(len(shortest_path[:-1]))]
            rate = 20000
            protocol = "TCP"
            rate = (min(all_capacities) * 10e3)/4
            rate = rate - rate/2

            max_rate, min_rate, mean = 0, 0, 0
            pattern = ""
            print(conn_info["pattern"])
            if conn_info["pattern"] == "uniform":
                rate = (min(all_capacities) * 10e3)/4
                min_rate = rate - rate/2
                max_rate = rate - 10000
                pattern = f" -U {min_rate} {max_rate} -u {packet_size} {2*packet_size}"
            elif conn_info["pattern"] == "congested":
                protocol = "UDP"
                rate = (min(all_capacities) * 10e3)/4
                rate = rate - rate/2
                pattern = f"-C {rate}"
            elif conn_info["pattern"] == "normal":
                mean =  rate - rate/2
                pattern = f"-C {mean} -n {packet_size} {0.3*packet_size}"
            elif conn_info["pattern"] == "exp":
                mean = rate - rate/2
                pattern = f"-E {mean} -e {packet_size}"
            elif conn_info["pattern"] == "poisson":
                mean = rate - rate/2
                pattern = f"-O {mean} -o {packet_size}"
            elif conn_info["pattern"] == "pareto":
                mean = rate - rate/2
                pattern = f"-C {mean} -v 1 {packet_size}"
            elif conn_info["pattern"] == "burst":
                mean = rate - rate/2
                pattern = f"-B O {mean} O {mean} -o {packet_size}"
            elif conn_info["pattern"] == "gamma":
                pattern = f"-G 0.5 {mean} -g 0.5 {packet_size}"
            else:
                raise ValueError("Pattern did not found!")

            duration = conn_info["duration"]
            mn_src_host.cmd(f"ITGSend -T {protocol} \
                                        -a 10.0.{dst_host // 256}.{dst_host % 256} \
                                        {pattern} \
                                        -t {duration}000 \
                                        -x {experiment_dir}/{conn_id}_traffic_results_rx.log &")

            info(f"Saving flow paths of connection {conn_id}\n")
            sleep(4)
            self.get_flow_paths(conn_info, conn_id, experiment_dir)
        sleep(duration)


    def get_flow_paths(self, conn_info,
                            conn_id,
                            experiment_dir: str = "./logs"
                            ):
        """
        Get the paths of a traffic flow
        """

        current_path = {}
        # get source and destination of a connection
        src_host = conn_info["src"] + 1
        dst_host = conn_info["dst"] + 1

        # convert to 16-char hexadecimal URI
        # using by ONOS to identify a device
        src_host_id = f"{src_host:016x}"
        dst_host_id = f"{dst_host:016x}"

        try:
            paths = requests.get(
                self.onos_api_adress + f"/paths/of%3A{src_host_id}/of%3A{dst_host_id}",
                    timeout=10,
                    auth=(self.onos_user, self.onos_pass)
            )
        except requests.exceptions.Timeout:
            print("ERROR: The requests time out")

        if paths.status_code == 200 and len(paths.json()["paths"]) != 0:
            all_paths = paths.json()["paths"]
            # the "minus 1" below is because the difference between
            # the node id on gml file that start with 0, whereas
            # openflow/onos adopting an id staring with 1. Anyway
            # tensorflow tools adopt 0. So this change is necessary
            current_path[conn_id] = [
                (int(all_paths[0]["links"][link_id]["src"]["device"].split(":")[1], 16) - 1,
                int(all_paths[0]["links"][link_id]["dst"]["device"].split(":")[1], 16) - 1)
                                    for link_id in range(len(all_paths[0]["links"]))
            ]
        else:
            os.system("sudo mn -c") # remove all switches and links
            raise Exception("The flow paths has been not collected")

        all_traffic_metadata = {}
        flowpath_filename = f"{experiment_dir}/mininet_data.json"
        if os.path.isfile(flowpath_filename):
            with open(flowpath_filename, "r", encoding="utf-8") as f:
                all_traffic_metadata = json.load(f)
        if "paths" in all_traffic_metadata.keys():
            all_traffic_metadata["paths"].update(current_path)
        else:
            all_traffic_metadata["paths"] = current_path
        # Writing paths to external file
        with open(flowpath_filename, "w", encoding="utf-8") as f:
            json.dump(all_traffic_metadata, f, indent=4)
