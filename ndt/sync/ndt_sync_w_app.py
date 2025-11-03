"""
Virtual twin with drift detection main script
and SLA monitoring use case
Created by: Cláudio Modesto
LASSE
"""

import os
from time import sleep
import pathlib
import argparse
import subprocess
import numpy as np
from river import drift
import tensorflow as tf
from ndt_synchronization import (predicting_vtwin,
                                    initial_training_vtwin,
                                    load_untrained_model,
                                    load_trained_model)

model_version = 0
async_running = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel('ERROR')

def main_loop(realization: int, target: str, data_dir, topology, sync):
    """
    Main loop traffic generation function
    """
    global model_version
    global async_running

    # used only to accelarate the simulation
    # when a retraining process is not running
    convey_time = 0
    root_data_dir = data_dir # root directory for the database
    if topology == "5g_crosshaul":
        dataset_name = "experiment_10"
        window_size = 6800
    elif topology == "germany":
        window_size = 7000
        dataset_name = "experiment_20"
    elif topology == "passion":
        dataset_name = "experiment_30"
        window_size = 6800
    else:
        raise ValueError("This is not a supported topology!")

    # Create the output directory
    output_path_name = f"results/{topology}/"
    if os.path.isfile(output_path_name):
        os.remove(output_path_name)
    else:
        pathlib.Path(output_path_name).mkdir(parents=True, exist_ok=True)

    # Create the weights directory
    weights_path_name = f"weights/{topology}/"
    if os.path.isfile(weights_path_name):
        os.remove(weights_path_name)
    else:
        pathlib.Path(weights_path_name).mkdir(parents=True, exist_ok=True)

    model_version = 0
    training_data = [f"{root_data_dir}/{topology}/{dataset_name}0_cv",
                    f"{root_data_dir}/{topology}/{dataset_name}1_cv",
                    f"{root_data_dir}/{topology}/{dataset_name}2_cv",
                    f"{root_data_dir}/{topology}/{dataset_name}4_cv"]

    model_weights_dir = f"weights/{topology}/model_version"

    if not os.path.isfile(f"{model_weights_dir}_{model_version}/final_weight.index"):
        untrained_model = load_untrained_model(target)
        initial_training_vtwin(training_data[model_version],
                                            untrained_model,
                                            model_weights_dir,
                                            realization=realization,
                                            topology=topology,
                                            target=target)

    stream_data_dir = [f"{root_data_dir}/{topology}/{dataset_name}0_cv/testing",
                        f"{root_data_dir}/{topology}/{dataset_name}1_cv/testing",
                        f"{root_data_dir}/{topology}/{dataset_name}2_cv/testing",
                        f"{root_data_dir}/{topology}/{dataset_name}4_cv/testing"]

    weight_filename = f"{model_weights_dir}_{model_version}/final_weight.index"
    last_weight_id = os.path.getmtime(weight_filename)
    stream_data = tf.data.Dataset.load(f"{stream_data_dir[0]}", compression="GZIP")

    for stream_dir in stream_data_dir[1:]:
        new_data = tf.data.Dataset.load(stream_dir,
                                                compression="GZIP")
        stream_data = stream_data.concatenate(new_data)

    trained_model = load_trained_model(training_data[model_version],
                                        f"{model_weights_dir}_{model_version}/final_weight",
                                        target)
    indexes, points = [], []
    flow_id = 0

    kswin = drift.KSWIN(alpha=0.001, window_size=window_size, stat_size=1200, seed=42)
    model_training = None
    window_index = 0
    model_updated = []
    drift_detected = []
    gt_all_sla_violations = []
    pred_all_sla_violations = []
    all_correct_pred = []

    for _, (sample_features, labels) in enumerate(stream_data):
        pred_sla_violations = 0
        gt_sla_violations = 0
        correct_pred = 0
        stream_data_sample = (sample_features, labels)
        weight_filename = f"{model_weights_dir}_{model_version}/final_weight.index"

        # update virtual twin model
        if (os.path.isfile(weight_filename) and \
            os.path.getmtime(weight_filename) > last_weight_id \
                                        and sync \
                                        and not async_running):
            print("New model")
            model_updated.append(window_index)
            last_weight_id = os.path.getmtime(weight_filename)
            print(f"Model version {model_version} in production")
            trained_model = load_trained_model(training_data[model_version],
                                    f"{model_weights_dir}_{model_version}/final_weight",
                                    target)

        predicted_delay, _ = predicting_vtwin(trained_model, stream_data_sample)

        for i, budget in enumerate((sample_features["flow_delay_budget"].numpy())):
            if predicted_delay[i] > budget:
                pred_sla_violations += 1
            if labels.numpy()[i] > budget:
                gt_sla_violations += 1
            if (predicted_delay[i] > budget) == (labels.numpy()[i] > budget):
                correct_pred += 1

        pred_all_sla_violations.append(pred_sla_violations)
        gt_all_sla_violations.append(gt_sla_violations)
        all_correct_pred.append(correct_pred)

        for flow_traffic in sample_features["flow_traffic"].numpy():
            print(flow_id)
            kswin.update(flow_traffic)
            if sync and kswin.drift_detected and not async_running:
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
                                "--topology", f"{topology}"])
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
    print("Saving Error Metrics")
    with open(f"results/{topology}/uc_violations_{sync}_r_{realization}.npz", "wb") as f:
        np.savez(f, pred_all_sla_violations, gt_all_sla_violations,
                            all_correct_pred, drift_detected, model_updated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology", "-t", help="Type of topology to be used in the experiments.", 
        type=str, required=True
    )
    parser.add_argument(
        "--realization", "-r", help="Number of realization to be performed.", 
        type=int, required=True
    )
    parser.add_argument(
        "--dir", "-d", help="Path to network traffic datasets.", 
        type=str, required=True
    )
    parser.add_argument(
        "--sync", "-s", help="Enable twin synchronization.", 
        action="store_true", required=False
    )

    args = parser.parse_args()

    for realization in range(args.realization):
        main_loop(realization, "delay", args.dir,
                                args.topology, args.sync)
