import random
import numpy as np
import networkx as nx

def generate_topology(net_size, graph_file):
    """
    Function to generate random topologies with different sizes
    """
    G = nx.Graph()

    # Set the maximum number of ToS that will use the input traffic of the network
    G.graph["label"] = f"synthetic_{net_size}"

    nodes = []
    node_degree = []
    for n in range(net_size):
        node_degree.append(random.choices([2,3,4,5,6],weights=[0.34,0.35,0.2,0.1,0.01])[0])

        nodes.append(n)
        G.add_node(n)

    finish = False
    while True:
        aux_nodes = list(nodes)
        n0 = random.choice(aux_nodes)
        aux_nodes.remove(n0)
        # Remove adjacents nodes (only one link between two nodes)
        for n1 in G[n0]:
            if n1 in aux_nodes:
                aux_nodes.remove(n1)
        if len(aux_nodes) == 0:
            # No more links can be added to this node - can not acomplish node_degree for this node
            nodes.remove(n0)
            if len(nodes) == 1:
                break
            continue
        n1 = random.choice(aux_nodes)
        G.add_edge(n0, n1)
        # Assign the link capacity to the link
        capacities_options = [100, 200, 220, 400, 150, 300, 240, 310, 370]
        propag_delay_options = [1.2 , 1.1 , 4.5, 3.2, 1.8, 2.4, 2.5, 5.1, 1.4, 0.5, 0.9]
        G[n0][n1]["capacity"] = int(np.random.choice(capacities_options))
        G[n0][n1]["delay"] = np.random.choice(propag_delay_options)

        for n in [n0,n1]:
            node_degree[n] -= 1
            if node_degree[n] == 0:
                nodes.remove(n)
                if len(nodes) == 1:
                    finish = True
                    break
        if finish:
            break

    edge_id = 0
    for edge in list(G.edges()):
        G[edge[0]][edge[1]]["id"] = edge_id
        edge_id += 1

    if not nx.is_connected(G):
        G = generate_topology(net_size, graph_file)
        return G

    nx.write_gml(G, graph_file)

net_size = 128
generate_topology(net_size, f"synth_topology_{net_size}.gml")
