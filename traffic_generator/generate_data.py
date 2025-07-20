'''
Generate deep learning input-output 
dataset from traffic generator data
Created by Cláudio Modesto
'''

import json
import glob
import copy
import argparse
import numpy as np
import networkx as nx
import tensorflow as tf

def _get_network_data(experiment_path, exp_file):
    '''
    Generate input-output data from data generator data
    '''
    with open(f'{experiment_path}/mininet_data.json', 'r', encoding='utf-8') as f:
        flow_paths = json.load(f)['paths']
    with open(f'{experiment_path}/mininet_data.json', 'r', encoding='utf-8') as f:
        packet_size = json.load(f)['tr_metadata']['tr_metadata']['packet_size']
    # load topology used in experiment
    topology = nx.read_gml(f'{experiment_path}/topology.gml')

    flows = []
    exp_conn_id = int(exp_file.split('/')[-1].split('_')[0])
    with open(exp_file, "r", encoding='utf-8') as file:
        for line in file:
            numbers = line.rstrip("\n").split(" ")  # Extract all numbers
            if float(numbers[1]) == 0 or float(numbers[4]) < 0:
                continue
            flows.append([float(feature) for feature in numbers] + [packet_size] + [exp_conn_id])

        # 1: bandwidth
        # 4: packet loss
        # -1: connection ID
        # -2: packet size
        # 2: delay
        # 3: jitter
        filtered_flow_param = np.take(np.array(flows),
                                    [1, 4, 3, -2, -1, 2], axis=1)
        filtered_flow_param[:, 0] = filtered_flow_param[:, 0] # convert to bits/s

    link_to_flow = []
    # get the links used for each flows
    for conn_id in filtered_flow_param[:, -2]:
        link_to_flow.append([topology.get_edge_data(
                        str(node[0]), str(node[1]))[0].get('id')
                    for node in flow_paths[str(int(conn_id))]])

    propag_delay = []
    # get propagation delay for each link
    for conn_id in filtered_flow_param[:, -2]:
        propag_delay.append([topology.get_edge_data(
                        str(node[0]), str(node[1]))[0].get('delay')
                    for node in flow_paths[str(int(conn_id))]])
    # get propagation delay per-flow
    propag_delay = np.sum(propag_delay, axis=1) / 10e3 # convert to s
    delay_budget = np.full(len(propag_delay), 
                        np.mean(filtered_flow_param[:, -1]) * 0.4 + np.mean(filtered_flow_param[:, -1]))

    # get capacity of used links
    capacity = [topology.get_edge_data(
                        str(node[0]), str(node[1]))[0].get('capacity')
                    for node in flow_paths[str(int(exp_conn_id))]]
    capacity = np.array(capacity) * 10e3 # convert to kbits/s
    # get the flows used for each links
    flow_to_link = []
    for link_id in range(len(topology.edges())):
        flow_ids = [(flow_id, 0) for flow_id in range(len(link_to_flow))
                            if link_id in link_to_flow[flow_id]]
        if len(flow_ids) == 0:
            continue

        flow_to_link.append(np.array(flow_ids))

    n_flows = len(link_to_flow)
    aux_link_to_flow = copy.deepcopy(link_to_flow)
    link_to_flow = []
    for i in range(n_flows):
        link_to_flow.append([i for i in range(len(aux_link_to_flow[i]))])

    # reshaping features based on batch size
    batch_size = 100
    remainder = len(filtered_flow_param) % batch_size
    if remainder != 0:
        filtered_flow_param = filtered_flow_param[:-remainder]
        propag_delay = propag_delay[:-remainder]
        delay_budget = delay_budget[:-remainder]
        link_to_flow = link_to_flow[:-remainder]

    filtered_flow_param = np.reshape(filtered_flow_param,
                                (int(filtered_flow_param.shape[0]/batch_size),
                                batch_size,
                                filtered_flow_param.shape[-1]))

    propag_delay = np.reshape(propag_delay, (int(propag_delay.shape[0]/batch_size),
                                batch_size))

    delay_budget = np.reshape(delay_budget, (int(delay_budget.shape[0]/batch_size),
                                batch_size))

    link_to_flow = np.array(link_to_flow)
    link_to_flow = np.reshape(link_to_flow, (int(link_to_flow.shape[0]/batch_size),
                                            batch_size, link_to_flow.shape[-1]))

    aux_flow_to_link = np.array(flow_to_link)
    flow_to_link = []
    for _ in range(batch_size, aux_flow_to_link.shape[1] + 1, batch_size):
        flow_to_link.append([[[j, 0] for j in range(batch_size)] * len(capacity)])
    flow_to_link = np.array(flow_to_link)
    flow_to_link = flow_to_link.reshape(flow_to_link.shape[0],
                                        len(capacity),
                                        batch_size,
                                        flow_to_link.shape[-1])
    samples = []
    for i in range(filtered_flow_param.shape[0]):
        if any(x == 0 for x in filtered_flow_param[i, :, -1] * 1000):
            continue
        sample = (
            {
                # identifier features
                "flow_traffic": np.expand_dims(
                    filtered_flow_param[i, :, 0], axis=1
                ),
                "flow_loss_packet": np.expand_dims(
                    filtered_flow_param[i, :, 1], axis=1
                ),
                "jitter": np.expand_dims(
                    filtered_flow_param[i, :, 2], axis=1
                ),
                "flow_packet_size": np.expand_dims(
                    filtered_flow_param[i, :, 3], axis=1
                ),
                "flow_propag_delay": np.expand_dims(
                    propag_delay[i, :], axis=1
                ),
                "flow_delay_budget": np.expand_dims(
                    delay_budget[i, :], axis=1
                ),

                "flow_length": np.expand_dims(
                    [len(flow) for flow in link_to_flow[i]], axis=1),
                # link attributes
                "link_capacity": np.expand_dims(
                    capacity, axis=1
                ),
                # topology attributes
                "link_to_flow": np.array(link_to_flow[i]),
                "flow_to_link": np.array(flow_to_link[i]),
            },
            filtered_flow_param[i, :, -1], # flow delay
        )
        samples.append(sample)

    return samples

def _generator(experiment_path):
    experiment_path = experiment_path.decode('utf-8')
    for exp_file in glob.glob(f'{experiment_path}/*_metric_results.txt'):
        print(exp_file)
        samples = _get_network_data(experiment_path, exp_file)
        for i, _ in enumerate(samples):
            yield samples[i]


def generate_tf_data(experiment_path):
    '''
    Convert numpy data to tensorflow data structure
    '''
    signature = (
        {
            "flow_traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_loss_packet": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "jitter": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_packet_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_propag_delay": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_delay_budget": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "flow_length": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "link_to_flow": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            "flow_to_link": tf.TensorSpec(shape=(None, None, 2), dtype=tf.int32),
        },
        tf.TensorSpec(shape=None, dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(
        _generator,
        args=[experiment_path],
        output_signature=signature,
    )

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the traffic generator")
    parser.add_argument("--exp-path",
                        type=str,
                        required=True,
                        help="Path to experiment files")
    parser.add_argument("--output-dir",
                        type=str,
                        required=True,
                        help="Path to output directory")
    args = parser.parse_args()

    # saving tensorflow input-output data
    tf.data.Dataset.save(
    generate_tf_data(
        args.exp_path,
    ),
    args.output_dir,
    compression="GZIP",
    )
    N_FOLDS = 20
    # Split dataset into n folds
    ds = tf.data.Dataset.load(args.output_dir, compression="GZIP")
    ds_split = [ds.shard(N_FOLDS, ii) for ii in range(N_FOLDS)]
    dataset_name = (
        args.output_dir if args.output_dir[-1] != "/" else args.output_dir[:-1]
    )

    VAL_IDX = 1
    TEST_IDX = 0
    # Split dataset into train and validation
    ds_val = ds_split[VAL_IDX]
    ds_test = ds_split[TEST_IDX]

    tr_splits = [jj for jj in range(N_FOLDS)]

    ds_train = ds_split[2]
    for jj in tr_splits[3:]:
        ds_train = ds_train.concatenate(ds_split[jj])
    # Save datasets
    tf.data.Dataset.save(
        ds_train, f"{dataset_name}_cv/training", compression="GZIP"
    )
    tf.data.Dataset.save(
        ds_val, f"{dataset_name}_cv/validation", compression="GZIP"
    )
    tf.data.Dataset.save(
        ds_test, f"{dataset_name}_cv/testing", compression="GZIP"
    )
    