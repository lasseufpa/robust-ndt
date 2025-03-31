#!/usr/bin/python

import os
import json
from time import sleep
from typing import Optional

from mininet.net import Mininet, CLI
from mininet.node import RemoteController
from mininet.node import Host
from mininet.node import OVSKernelSwitch
from mininet.log import info
from mininet.link import TCLink
import networkx as nx
import requests


class Network:
    '''
    Define mininet topology methods and attributes 
    '''
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
        '''
        Instatiate a miniet topology and generate traffic in it
        '''
        self.net.start()
        info("INFO: Mininet topology started\n")
        sleep(time_wait_topology)
        info("INFO: Traffic generator started\n")
        self.start_servers(flows_description, experiment_dir)
        self.start_clients(flows_description, experiment_dir)
        info(f"INFO: Experiment is finishing, wait a second. You can find log information in {experiment_dir}\n")
        sleep(10)
        self.net.stop()

    def gen_mac_address(self, idx: int):
        return f"00:00:00:00:00:{idx:02x}"

    def get_device_id(self, idx: int):
        return f"of:{(idx+1):016d}"

    def start_servers(self, flows_description: dict,
                        experiment_dir: str = "./logs"):
        for _, conn_info in flows_description.items():
            mn_dst_host = self.hosts[conn_info["dst"]]
            conn_id = conn_info['conn_id']
            mn_dst_host.cmd(f"iperf -s -e -u \
                            -p 5001 -e -i 1 \
                            -x CVMS \
                            -o {experiment_dir}/{conn_id}_traffic_results.txt &")

    def start_clients(self,
                    flows_description: dict,
                    experiment_dir: str = "./logs"):

        flowpath_filename = f"{experiment_dir}/conn_paths.json"
        if os.path.isfile(flowpath_filename):
            os.remove(flowpath_filename)

        duration = 0
        for _, conn_info in flows_description.items():
            mn_host_src = self.hosts[conn_info["src"]]
            dst_host = conn_info["dst"] + 1
            duration = conn_info["duration"]
            bandwidth = conn_info["bandwidth"]
            conn_id = conn_info['conn_id']
            n_streams = conn_info["n_streams"]
            mn_host_src.cmd(f'nohup iperf -e \
                                            -c 10.0.0.{dst_host} \
                                            -p 5001 -u \
                                            -b {bandwidth}M -i 1 \
                                            -t {duration} \
                                            -P {n_streams} &')
            info(f'Saving flow paths of connection {conn_id}\n')
            sleep(10)
            self.get_flow_paths(conn_info, conn_id, experiment_dir)
        sleep(duration + 100)

    def get_flow_paths(self, conn_info,
                            conn_id,
                            experiment_dir: str = "./logs"
                            ):
        '''
        Get the paths of a traffic flow
        '''

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
            print('ERROR: The requests time out')

        if paths.status_code == 200 and len(paths.json()['paths']) != 0:
            all_paths = paths.json()['paths']
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
            raise Exception('The flow paths has been not collected')

        flowpath_filename = f"{experiment_dir}/conn_paths.json"
        if os.path.isfile(flowpath_filename):
            with open(flowpath_filename, "r", encoding='utf-8') as f:
                flow_paths = json.load(f)
            flow_paths.update(current_path)
        else:
            flow_paths = current_path
        # Writing paths to external file
        with open(flowpath_filename, "w", encoding='utf-8') as f:
            json.dump(flow_paths, f, indent=4)
