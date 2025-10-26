"""
Operational NDT main script
Created by: Cláudio Modesto
"""

import os
from time import sleep
import pathlib
import argparse
import subprocess
import numpy as np
from river import drift
from std_train import get_mean_std_dict, train_and_evaluate, get_default_hyperparams
import tensorflow as tf
import std_delay_model


model_version = 0
async_running = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--topology", "-t", help="Type of topology to be used in the experiments.", 
    type=str, required=True
)
parser.add_argument(
    "--dir", "-d", help="Path to network traffic datasets.", 
    type=str, required=True
)
parser.add_argument(
    "--realization", "-r", help="Number of realization to be performed.", 
    type=int, required=True
)
parser.add_argument(
    "--sync", "-s", help="Enable twin synchronization.", 
    action="store_true", required=False
)

args = parser.parse_args()

def main_loop(realization: int):
    """
    Main loop traffic generation function
    """
    global model_version
    global async_running

    # used only to accelarate the simulation
    # when a retraining process is not running
    convey_time = 0
    root_data_dir = args.dir # root directory for the database
    if args.topology == "5g_crosshaul":
        dataset_name = "experiment_10"
        window_size = 6800
    elif args.topology == "germany":
        window_size = 7000
        dataset_name = "experiment_20"
    elif args.topology == "passion":
        dataset_name = "experiment_30"
        window_size = 6800
    else:
        raise ValueError("This is not a supported topology!")

    # Create the output directory
    output_path_name = f"results/{args.topology}/"
    if os.path.isfile(output_path_name):
        os.remove(output_path_name)
    else:
        pathlib.Path(output_path_name).mkdir(parents=True, exist_ok=True)

    # Create the weights directory
    weights_path_name = f"weights/{args.topology}/"
    if os.path.isfile(weights_path_name):
        os.remove(weights_path_name)
    else:
        pathlib.Path(weights_path_name).mkdir(parents=True, exist_ok=True)

    model_version = 0
    training_data = [f"{root_data_dir}/{args.topology}/{dataset_name}0_cv",
                    f"{root_data_dir}/{args.topology}/{dataset_name}1_cv",
                    f"{root_data_dir}/{args.topology}/{dataset_name}2_cv",
                    f"{root_data_dir}/{args.topology}/{dataset_name}4_cv"]

    model_weights_dir = f"weights/{args.topology}/model_version"

    if not os.path.isfile(f"{model_weights_dir}_{model_version}/final_weight.index"):
        print("Training the initial model!!")
        untrained_model = load_untrained_model()
        initial_training_vtwin(training_data[model_version],
                                    untrained_model,
                                    model_weights_dir,
                                    topology=args.topology,
                                    realization=realization)

    stream_data_dir = [f"{root_data_dir}/{args.topology}/{dataset_name}0_cv/testing",
                        f"{root_data_dir}/{args.topology}/{dataset_name}1_cv/testing",
                        f"{root_data_dir}/{args.topology}/{dataset_name}2_cv/testing",
                        f"{root_data_dir}/{args.topology}/{dataset_name}4_cv/testing"]

    weight_filename = f"{model_weights_dir}_{model_version}/final_weight.index"
    last_weight_id = os.path.getmtime(weight_filename)
    stream_data = tf.data.Dataset.load(f"{stream_data_dir[0]}", compression="GZIP")

    for stream_dir in stream_data_dir[1:]:
        new_data = tf.data.Dataset.load(stream_dir,
                                                compression="GZIP")
        stream_data = stream_data.concatenate(new_data)

    trained_model = load_trained_model(training_data[model_version],
                                        f"{model_weights_dir}_{model_version}/final_weight")
    nmses, indexes, points = [], [], []
    flow_id = 0

    # defining the concept drift detector
    kswin = drift.KSWIN(alpha=0.001, window_size=window_size, stat_size=1200, seed=42)
    model_training = None
    window_index = 0
    model_updated = []
    drift_detected = []
    for _, (sample_features, labels) in enumerate(stream_data):
        stream_data_sample = (sample_features, labels)
        weight_filename = f"{model_weights_dir}_{model_version}/final_weight.index"

        # update virtual twin model
        if (os.path.isfile(weight_filename) and \
            os.path.getmtime(weight_filename) > last_weight_id \
                                        and args.sync \
                                        and not async_running):
            print("New model")
            model_updated.append(window_index)
            last_weight_id = os.path.getmtime(weight_filename)
            print(f"Model version {model_version} in production")
            trained_model = load_trained_model(training_data[model_version],
                                    f"{model_weights_dir}_{model_version}/final_weight")

        _, nmse = predicting_vtwin(trained_model, stream_data_sample)
        nmses.append(nmse)

        for flow_traffic in sample_features["flow_traffic"].numpy():
            print(flow_id)
            kswin.update(flow_traffic)
            if args.sync and kswin.drift_detected and not async_running:
                drift_detected.append(window_index)
                if model_version + 2 > len(training_data):
                    break
                print("DRIFT detected")
                convey_time = 0.8
                indexes.append(flow_id)
                points.append(sample_features["flow_traffic"].numpy())
                print("Retraing the model")
                async_running = True
                model_version += 1
                model_training = subprocess.Popen(["python3", "std_train.py",
                                "--ds-train", f"{training_data[model_version]}",
                                "--ckpt-path", f"{model_weights_dir}_{model_version}",
                                "--topology", f"{args.topology}",
                                "--realization", f"{realization}"])
            flow_id += 1
            if flow_id > window_size - 200:
                sleep(convey_time)
            if async_running and model_training.poll() is not None:
                print("Retraining is over")
                convey_time = 0 # accelerating the simulation when a retraining isn't running
                async_running = False
        window_index += 1
    if model_training is not None:
        model_training.kill()
    print("Saving error Metrics")
    np.savez(f"{output_path_name}/results_sync_{args.sync}_r_{realization}.npz",
                                np.array(nmses), drift_detected, model_updated)

def load_untrained_model():
    """
    function to load model without training weights
    """
    model = std_delay_model.VirtualTwin

    return model


def load_trained_model(training_data, model_weights_file):
    """
    function to load a trained GNN model
    """
    model = std_delay_model.VirtualTwin()

    model.load_weights(f"{model_weights_file}")

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        metrics=tf.keras.metrics.MeanAbsolutePercentageError()
    )

    model.set_mean_std_scores(
        get_mean_std_dict(
            tf.data.Dataset.load(f"{training_data}/training",
            compression="GZIP"),
            model.mean_std_scores_fields,
        )
    )

    return model


def predicting_vtwin(trained_model, stream_data):
    """
    function to prediction per-flow delay using
    a trained GNN model
    """
    # obtain the prediction as numpy array, and flatten
    predicted_delay = trained_model(stream_data[0]).numpy().reshape((-1,))
    ground_truth_delay = stream_data[1].numpy()
    err_metric = np.mean((ground_truth_delay - predicted_delay)**2)/np.mean(ground_truth_delay**2)
    err_metric = 10*np.log10(err_metric)

    return predicted_delay, err_metric


def initial_training_vtwin(training_data, model,
                                model_weights_dir,
                                topology,
                                realization):
    """
    function to training GNN model
    """
    global async_running
    global model_version

    model_weights_dir = f"{model_weights_dir}_{model_version}/"
    train_and_evaluate(
        os.path.join(training_data),
        model(),''
        **get_default_hyperparams(),
        ckpt_path=model_weights_dir,
        topology=topology,
        realization=realization
    )

if __name__ == "__main__":
    N_REALIZATIONS = args.realization

    for realization in range(N_REALIZATIONS):
        main_loop(realization)
